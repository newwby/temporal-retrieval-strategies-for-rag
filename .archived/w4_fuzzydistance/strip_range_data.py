# relative path from ./src/w4_fuzzydistance/strip_ranges.py
import json
import os
from pathlib import Path
from unittest import result

import urllib
from consts import INVALID_RAW_DATA_DIRECTORY, MONTHS, OLLAMA_ENABLED, RAW_DATA_DIRECTORY, PROCESSED_DATA_PATH, EXTRACTED_DATA_FILE_NAME, DEBUG_MODE, SIMPLIFIED_DATA_FILE_NAME, UNDEFINED_KEY

"""
Strip range data from the raw data .jsonl files and output to a single .jsonl file.
This output is the authorative file for ID lookups (buckets & embedding files will just reference ID for lookup from this file)
"""

# when called from w4main the access function should be strip_ranges()
__all__ = ['strip_ranges']


"""
function to group ranges into deterministic fuzzy categories based on the start and end years, and calculate distance and midpoint
"""
#TODO add buckets for data (grouping by IDX) for entries within decades/centuries?
#TODO experiment with alternative fuzzy phrasing
def _group_ranges(entry):
    start_month = entry["start_month"] if entry["start_month"] is not None else 1
    end_month = entry["end_month"] if entry["end_month"] is not None else 12
    start_year = int(entry["start_year"])
    end_year = int(entry["end_year"])
    
    # Grouping logic: for now, just group by decade, ignoring months.
    # TODO update fuzzy logic to include months, e.g. early/mid/late decade, early/mid/late month, etc.
    start_decade = start_year // 10 * 10
    end_decade = end_year // 10 * 10
    fuzzy_year_start = "early" if start_year % 10 <= 3 else "mid" if start_year % 10 <= 6 else "late"
    fuzzy_year_end = "early" if end_year % 10 <= 3 else "mid" if end_year % 10 <= 6 else "late"
    
    # fuzzy range created from above logic, e.g. "early 1930s to late 1940s"
    range_fuzzy = f"{fuzzy_year_start} {start_decade}s to {fuzzy_year_end} {end_decade}s"

    return {
        "fuzzy_year_start": fuzzy_year_start,
        "fuzzy_year_end": fuzzy_year_end,
        "start_decade": start_decade,
        "end_decade": end_decade,
        "range_fuzzy": range_fuzzy
    }


"""
Write the calculated distance and midpoint for an entry, using months as a unit
IMPORTANT - where months are null Jan is assumed for start and Dec is assumed for end, so that distance can be calculated as 12 months for a year-only range
"""
def _write_distance(entry):
    distance = 0
    start_month = entry["start_month"] if entry["start_month"] is not None else 1
    end_month = entry["end_month"] if entry["end_month"] is not None else 12
    start_year = int(entry["start_year"])
    end_year = int(entry["end_year"])
    # calculate distance in months
    month_distance = end_month - start_month
    year_distance_in_months = (end_year - start_year) * 12
    distance = month_distance + year_distance_in_months
    midpoint_year = start_year + (end_year - start_year) // 2
    midpoint_month = start_month + (end_month - start_month) // 2

    return {"distance": distance, "midpoint_year": midpoint_year, "midpoint_month": midpoint_month}


"""
module to strip range data from the raw data .jsonl files (output by strip_questions.py) and output to a single .jsonl file
"""
def _extract_range_data(obj, error_string):
    # extract range bounds from the raw data
    range_data = {}

    # different handling for years vs months
    # first confirm the keys are found in the entry
    expected_input_keys = ["start_precision", "end_precision", "start_text", "end_text"]
    input_is_valid = [key for key in expected_input_keys if key in obj]
    if len(input_is_valid) != len(expected_input_keys):
        if DEBUG_MODE:
            print(f"{error_string} has incomplete input data")
        # returning empty will cause the entry to be skipped
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
           # returning empty will cause the entry to be skipped
            return {}
        
        range_data["start_month"] = _get_bounds(obj, start_bounds, is_start=True)["month"]
        range_data["start_year"] = _get_bounds(obj, start_bounds, is_start=True)["year"]
        range_data["end_month"] = _get_bounds(obj, end_bounds, is_start=False)["month"]
        range_data["end_year"] = _get_bounds(obj, end_bounds, is_start=False)["year"]
        return range_data


"""
obj (dict): the jsonl entry to extract bounds from
precision (str): the precision of the bounds, either "year" or "month"
is_start (bool): True if the bounds are for the start of the range, False if for the end of the range
"""
def _get_bounds(obj, precision, is_start):
    bounds = {}
    key = "start_text" if is_start else "end_text"
    # if precision is year, text will read "Year" - just return the year
    if precision == "year":
        bounds["year"] = obj[key]
        bounds["month"] = None
    # if precision is month, text will read "Month Year" - split the text into month and year
    if precision == "month":
        text_split = [i.strip() for i in obj[key].split()]
        assert(len(text_split) == 2)
        bounds["year"] = text_split[1]
        # convert month text to number using MONTHS dict from consts.py
        bounds["month"] = MONTHS[text_split[0].lower()]
    return bounds


"""
Convert the range data into a natural language equivalent
    e.g. "Nov 1935 to Sep 1937" becomes "from November nineteen thirty-five to September nineteen thirty-seven"
(This is useful for embedding comparisons, as the natural language equivalent is more likely to match other text)
# start psh and run > ollama run llama3.1:8b
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
Convert the range data into a natural language equivalent
    e.g. "Nov 1935 to Sep 1937" becomes "from November nineteen thirty-five to September nineteen thirty-seven"
(This is useful for embedding comparisons, as the natural language equivalent is more likely to match other text)
# start psh and run > ollama run llama3.1:8b
"""
def _REMOVE_ME(range_numeric_text):
    # handling to be added for range_language_text specifying
    if not OLLAMA_ENABLED:
        if DEBUG_MODE:
            print(f"Ollama calls disabled, skipping natural language conversion for {range_numeric_text}")
        return UNDEFINED_KEY
    
    # errors outputting tbd instead of undefined to show when it ran vs when it didn't
    range_language_text = "tbd"

    prompt = f"""
    Convert the following range into a natural language equivalent.
    E.g. 'Nov 1935 to Sep 1937' becomes 'from November nineteen thirty-five to September nineteen thirty-seven'.
    IMPORTANT - do not add any additional commentary or explanation, return only the converted text.
    This output is for a jsonl field and will be used for embedding comparisons, so it should be as close to the original range as possible.
    Input to convert: {range_numeric_text}
    """
    
    url = "http://127.0.0.1:11434/api/generate"
    model = "llama3.1:8b"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    if DEBUG_MODE:
        print(f"Sending request to Ollama with model '{model}', converting {range_numeric_text}...")
    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not connect to Ollama. Is Ollama running on localhost:11434?"
        ) from e
    
    range_language_text = result["response"]
    
    if DEBUG_MODE:
        print(f"Converted '{range_numeric_text}' to '{range_language_text}'.")
    
    return range_language_text


"""
Expected data shape from strip_questions.py looks like this:
"range_numeric": "Nov 1935 to Sep 1937", "start_text": "Nov 1935", "end_text": "Sep 1937", "start_precision": "month", "end_precision": "month", 
"range_numeric": "2005 to 2007", "start_text": "2005", "end_text": "2007", "start_precision": "year", "end_precision": "year", 
"""
def _process_raw_json_data(target_path):
    # for later referencing need to keep a record of order; source starts at -1 because incremented at start of loop due to error exits
    source_idx = -1
    output_idx = 0
    output = {}

    for file_path in target_path.glob("*.jsonl"):
        # don't bother processing files without explicit ranges
        if INVALID_RAW_DATA_DIRECTORY in str(file_path):
            if DEBUG_MODE:
                print(f"skipping file with no explicit ranges: {file_path}")
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                # step 1, setup
                # always increment source to track, this is why it starts at -1
                source_idx += 1
                # get the jsonl entry & setup entry for this obj, always encode id into the value as well as (later) key
                obj = json.loads(line)
                entry = {"source_id": source_idx, "output_id": output_idx}
                
                # step 2, error handling
                # extract all appropriate values
                # first the original range text, to use as an error reference
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
                # combine existing entry with the extracted range data, if any
                # if no range data is extracted, skip the entry
                range_data = _extract_range_data(obj, error_string)
                if len(range_data) == 0:
                    if DEBUG_MODE:
                        print(f"{error_string} has no range data extracted")
                    continue
                else:
                    # after successful extraction perform distance calculation and grouping
                    entry = {**entry, **range_data}
                    entry = {**entry, **_write_distance(entry), **_group_ranges(entry)}

                    # add the natural language equivalent of the range for embedding comparison
                    print(f"Processing entry {source_idx}, output {output_idx}", flush=True)
                    entry["range_language"] = _get_natural_language_equivalent(entry["range_numeric"])

                # step 4, validation
                # confirm expected data shape and flag anything unexpected
                expected_output_keys = ["source_id", "output_id", "range_numeric", "start_month", "start_year", "end_month", "end_year"]
                output_is_valid = [key for key in expected_output_keys if key in entry]
                if len(output_is_valid) != len(expected_output_keys):
                    if DEBUG_MODE:
                        print(f"{error_string} has incorrect keys: {list(entry.keys())}")
                    continue

                # step 5, output
                # ignore files where no successful extraction happened
                if len(entry) > 1:
                    output[output_idx] = entry
                    output_idx += 1
                else:
                    if DEBUG_MODE:
                        print(f"cannot output entry {error_string}")
                    continue
    
    return output


"""
Remove duplicate range entries based on their start and end dates.
"""
# TODO - this does not renumber output_id, so if duplicates are removed the output_id will not be sequential.
#   solution would be to reassign output_id after duplicates (requires full data rebuild after doing so)
def _remove_duplicates(data):
    cleaned_data = {}
    skipped_entries = 0
    id_record = []
    for key in data:
        entry = data[key]
        unique_id = f"{entry["start_year"]}-{entry["start_month"]}-{entry["end_year"]}-{entry["end_month"]}"
        if unique_id in id_record:
            skipped_entries += 1
            continue
        else:
            cleaned_data[key] = data[key]
            id_record.append(unique_id)
    print(f"total unique entries after cleaning = {len(id_record)}")
    print(f"entries skipped due to duplicate ranges = {skipped_entries}")

    # for initial logging
    if DEBUG_MODE:
        unique_range_check = {}
        for key in data:
            entry = data[key]
            range_key = f"{entry["start_year"]}-{entry["start_month"]}-{entry["end_year"]}-{entry["end_month"]}"
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
#

"""
File not very readable, keep the authorative but strip it down to output_idx, range_numeric, range_fuzzy
Simplified data only contains fields necessary for embedding/comparison, and is easier to read as well
"""
def _simplify_range_data(data):
    allowed_keys = ["output_idx", "range_numeric", "range_fuzzy", "range_language"]
    new_data = data.copy()
    for key in data:
        entry = data[key]
        new_entry = {k: v for k, v in entry.items() if k in allowed_keys}
        new_data[key] = new_entry
    return new_data


def _write_file(file_output, output_path):
    # output if any data
    if len(file_output) > 0:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for key in file_output:
                # key already encoded
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
strip all ranges found in raw_data directory .jsonl files, output to a single .jsonl file each time (overwriting previous output)
need to check for and get specific fields from existing .jsonl files (that were output by strip_questions.py) only
"""
def strip_ranges():
    base_path = Path(__file__).resolve().parent
    target_path = base_path / RAW_DATA_DIRECTORY
    output_dir = base_path / PROCESSED_DATA_PATH
    output_path = output_dir / EXTRACTED_DATA_FILE_NAME
    simplified_output_path = output_dir / SIMPLIFIED_DATA_FILE_NAME

    # confirm directories exist, write blank output file to overwrite any previous output
    target_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    # build the output before writing as a single file
    range_data = _process_raw_json_data(target_path)
    #TODO move languagerange conversion to after duplicate removal, it is running on duplicates and slowing down the process
    cleaned_data = _remove_duplicates(range_data)
    # two copies - one full, one simplified for easier human reading & embedding comparison
    _write_file(cleaned_data, output_path)
    _write_file(_simplify_range_data(cleaned_data), simplified_output_path)


"""
No arguments necessary, always works over the raw_data directory and outputs to the processed_data directory, overwriting any previous output
"""
if __name__ == "__main__":
    strip_ranges()

