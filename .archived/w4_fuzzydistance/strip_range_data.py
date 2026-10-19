import json
from pathlib import Path

import urllib.error
import urllib.request
from consts import INVALID_RAW_DATA_DIRECTORY, MONTHS, OLLAMA_ENABLED, RAW_DATA_DIRECTORY, PROCESSED_DATA_PATH, EXTRACTED_DATA_FILE_NAME, DEBUG_MODE, SIMPLIFIED_DATA_FILE_NAME, UNDEFINED_KEY

"""
merges ranges from raw_data .jsonl files into one .jsonl file
this is the authorative file, other files only reference its ids
"""

# entry point when called from main.py
__all__ = ['strip_ranges']


"""
groups ranges into fuzzy categories by start and end year
"""
#TODO bucket entries by decade/century (group by idx)?
#TODO try other fuzzy phrasing
def _group_ranges(entry):
    start_month = entry["start_month"] if entry["start_month"] is not None else 1
    end_month = entry["end_month"] if entry["end_month"] is not None else 12
    start_year = int(entry["start_year"])
    end_year = int(entry["end_year"])
    
    # group by decade only, months ignored for now
    #TODO use months too, e.g. early/mid/late month
    start_decade = start_year // 10 * 10
    end_decade = end_year // 10 * 10
    fuzzy_year_start = "early" if start_year % 10 <= 3 else "mid" if start_year % 10 <= 6 else "late"
    fuzzy_year_end = "early" if end_year % 10 <= 3 else "mid" if end_year % 10 <= 6 else "late"
    
    # e.g. "early 1930s to late 1940s"
    range_fuzzy = f"{fuzzy_year_start} {start_decade}s to {fuzzy_year_end} {end_decade}s"

    return {
        "fuzzy_year_start": fuzzy_year_start,
        "fuzzy_year_end": fuzzy_year_end,
        "start_decade": start_decade,
        "end_decade": end_decade,
        "range_fuzzy": range_fuzzy
    }


"""
works out distance and midpoint in months
important - null months default to jan for start and dec for end
"""
def _write_distance(entry):
    distance = 0
    start_month = entry["start_month"] if entry["start_month"] is not None else 1
    end_month = entry["end_month"] if entry["end_month"] is not None else 12
    start_year = int(entry["start_year"])
    end_year = int(entry["end_year"])
    # distance in months
    month_distance = end_month - start_month
    year_distance_in_months = (end_year - start_year) * 12
    distance = month_distance + year_distance_in_months
    midpoint = start_year * 12 + (start_month - 1) + distance // 2
    midpoint_year = midpoint // 12
    midpoint_month = midpoint % 12 + 1

    return {"distance": distance, "midpoint_year": midpoint_year, "midpoint_month": midpoint_month}


"""
reads start/end month and year from a strip_questions.py entry
"""
def _extract_range_data(obj, error_string):
    range_data = {}

    # check keys exist, then handle years vs months
    expected_input_keys = ["start_precision", "end_precision", "start_text", "end_text"]
    input_is_valid = [key for key in expected_input_keys if key in obj]
    if len(input_is_valid) != len(expected_input_keys):
        if DEBUG_MODE:
            print(f"{error_string} has incomplete input data")
        # empty return skips the entry
        return {}
    if not ("start_precision" in obj) or not ("end_precision" in obj):
        if DEBUG_MODE:
            print(f"{error_string} has missing precision data")
        return {}
    else:
        start_bounds = obj["start_precision"]
        end_bounds = obj["end_precision"]

        if not (start_bounds in ["year", "month"]) or not (end_bounds in ["year", "month"]):
            if DEBUG_MODE:
                print(f"{error_string} has unexpected data shape - confirm precision: [{start_bounds} vs {end_bounds}]")
            # empty return skips the entry
            return {}
        
        range_data["start_month"] = _get_bounds(obj, start_bounds, is_start=True)["month"]
        range_data["start_year"] = _get_bounds(obj, start_bounds, is_start=True)["year"]
        range_data["end_month"] = _get_bounds(obj, end_bounds, is_start=False)["month"]
        range_data["end_year"] = _get_bounds(obj, end_bounds, is_start=False)["year"]
        return range_data


"""
obj (dict): jsonl entry
precision (str): "year" or "month"
is_start (bool): True for range start, False for end
"""
def _get_bounds(obj, precision, is_start):
    bounds = {}
    key = "start_text" if is_start else "end_text"
    # year precision, text is just the year
    if precision == "year":
        bounds["year"] = obj[key]
        bounds["month"] = None
    # month precision, text is "month year" so split
    if precision == "month":
        text_split = [i.strip() for i in obj[key].split()]
        assert(len(text_split) == 2)
        bounds["year"] = text_split[1]
        # month name to number via MONTHS
        bounds["month"] = MONTHS[text_split[0].lower()]
    return bounds


"""
converts a range to words via ollama, closer to normal text for embedding comparison
    e.g. "Nov 1935 to Sep 1937" -> "from November nineteen thirty-five to September nineteen thirty-seven"
run first in powershell: ollama run llama3.1:8b
"""
def _get_natural_language_equivalent(range_numeric_text):
    if not OLLAMA_ENABLED:
        return UNDEFINED_KEY

    prompt = (
        "Convert this date range into words. Return only the converted range.\n"
        "Example: Nov 1935 to Sep 1937 -> "
        "from November nineteen thirty-five to September nineteen thirty-seven\n"
        f"Range: {range_numeric_text}"
    )

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": 50,
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Converting: {range_numeric_text}", flush=True)

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out converting: {range_numeric_text}"
        ) from exc
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at http://127.0.0.1:11434"
        ) from exc

    converted = result["response"].strip()
    print(f"Result: {converted}", flush=True)
    return converted

"""
expected input from strip_questions.py, e.g.
"range_numeric": "Nov 1935 to Sep 1937", "start_text": "Nov 1935", "end_text": "Sep 1937", "start_precision": "month", "end_precision": "month",
"range_numeric": "2005 to 2007", "start_text": "2005", "end_text": "2007", "start_precision": "year", "end_precision": "year",
"""
def _process_raw_json_data(target_path):
    # track order for later lookups, source starts at -1 as it's incremented at loop start (before any skips)
    source_idx = -1
    output_idx = 0
    output = {}

    for file_path in target_path.glob("*.jsonl"):
        # skip files without explicit ranges
        if INVALID_RAW_DATA_DIRECTORY in str(file_path):
            if DEBUG_MODE:
                print(f"skipping file with no explicit ranges: {file_path}")
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                # step 1, setup
                # always increment, hence the -1 start
                source_idx += 1
                # id stored in the value as well as (later) the key
                obj = json.loads(line)
                entry = {"source_id": source_idx, "output_id": output_idx}
                
                # step 2, error handling
                # range text first, used as an error reference
                if "range_numeric" in obj:
                    entry["range_numeric"] = obj["range_numeric"]
                else:
                    if DEBUG_MODE:
                        print(f"{source_idx} - range text not found in obj")
                    continue
                # used to identify entries with errors
                range_error_text = "[no range found!]" if not "range_numeric" in entry else entry["range_numeric"]
                error_string = f"id {source_idx} (range: {range_error_text}) (length {len(entry)})"
                
                # step 3, data extraction
                # merge extracted range data into entry, skip if none
                range_data = _extract_range_data(obj, error_string)
                if len(range_data) == 0:
                    if DEBUG_MODE:
                        print(f"{error_string} has no range data extracted")
                    continue
                else:
                    # add distance and fuzzy grouping
                    entry = {**entry, **range_data}
                    entry = {**entry, **_write_distance(entry), **_group_ranges(entry)}

                    # add natural language version for embedding comparison
                    print(f"Processing entry {source_idx}, output {output_idx}", flush=True)
                    entry["range_language"] = _get_natural_language_equivalent(entry["range_numeric"])

                # step 4, validation
                # flag unexpected data shape
                expected_output_keys = ["source_id", "output_id", "range_numeric", "start_month", "start_year", "end_month", "end_year"]
                output_is_valid = [key for key in expected_output_keys if key in entry]
                if len(output_is_valid) != len(expected_output_keys):
                    if DEBUG_MODE:
                        print(f"{error_string} has incorrect keys: {list(entry.keys())}")
                    continue

                # step 5, output
                # skip entries with nothing extracted
                if len(entry) > 1:
                    output[output_idx] = entry
                    output_idx += 1
                else:
                    if DEBUG_MODE:
                        print(f"cannot output entry {error_string}")
                    continue
    
    return output


"""
removes entries with duplicate start and end dates
"""
#TODO output_id isn't renumbered so has gaps after removal
#   fix: reassign output_id after (needs full data rebuild)
def _remove_duplicates(data):
    cleaned_data = {}
    skipped_entries = 0
    id_record = []
    for key in data:
        entry = data[key]
        unique_id = f"{entry['start_year']}-{entry['start_month']}-{entry['end_year']}-{entry['end_month']}"
        if unique_id in id_record:
            skipped_entries += 1
            continue
        else:
            cleaned_data[key] = data[key]
            id_record.append(unique_id)
    print(f"total unique entries after cleaning = {len(id_record)}")
    print(f"entries skipped due to duplicate ranges = {skipped_entries}")

    # debug logging
    if DEBUG_MODE:
        unique_range_check = {}
        for key in data:
            entry = data[key]
            range_key = f"{entry['start_year']}-{entry['start_month']}-{entry['end_year']}-{entry['end_month']}"
            if range_key in unique_range_check.keys():
                count = unique_range_check[range_key]
                unique_range_check[range_key] = (count + 1)
            else:
                unique_range_check[range_key] = 1
        
        for key in unique_range_check:
            count = unique_range_check[key]
            print(key)
            print(unique_range_check[key])
            if count > 1:
                print(f"the combination {key} is a duplicate! Occurs {count} times!")
    
    return cleaned_data

"""
stripped down copy of the authorative data (output_id, range_numeric, range_fuzzy, range_language)
only fields needed for embedding/comparison, easier to read
"""
def _simplify_range_data(data):
    allowed_keys = ["output_id", "range_numeric", "range_fuzzy", "range_language"]
    new_data = data.copy()
    for key in data:
        entry = data[key]
        new_entry = {k: v for k, v in entry.items() if k in allowed_keys}
        new_data[key] = new_entry
    return new_data


def _write_file(file_output, output_path):
    # write if any data
    if len(file_output) > 0:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for key in file_output:
                # key already in value
                out_f.write(json.dumps(file_output[key], ensure_ascii=False) + "\n")
    else:
        print(f"output empty, cannot write to {output_path}")

    # done
    output_size = len(list(Path(output_path).open("r", encoding="utf-8")))    
    if output_size == 0:
        print("No ranges found in any input files.")
    else:
        print(f"Stripped ranges written to {output_path}, total lines: {output_size}")


"""
merges all ranges from raw_data .jsonl files (from strip_questions.py) into one .jsonl file
overwrites previous output each run
"""
def strip_ranges():
    base_path = Path(__file__).resolve().parent
    target_path = base_path / RAW_DATA_DIRECTORY
    output_dir = base_path / PROCESSED_DATA_PATH
    output_path = output_dir / EXTRACTED_DATA_FILE_NAME
    simplified_output_path = output_dir / SIMPLIFIED_DATA_FILE_NAME

    # make dirs, blank output files to overwrite old output
    target_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    simplified_output_path.write_text("", encoding="utf-8")

    # build all output before writing
    range_data = _process_raw_json_data(target_path)
    #TODO move language conversion after duplicate removal, it runs on duplicates and slows things down
    cleaned_data = _remove_duplicates(range_data)
    # two copies - full, and simplified for reading & embedding
    _write_file(cleaned_data, output_path)
    _write_file(_simplify_range_data(cleaned_data), simplified_output_path)


"""
no args, reads raw_data and overwrites processed_data output
"""
if __name__ == "__main__":
    strip_ranges()

