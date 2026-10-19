"""
week 4 experiment - compares embedding distance between explicit and fuzzy date ranges, using timeqa questions
self-contained module, this runs the other scripts in order

expected layout (from repo root):
    data/
        timeqa/
            dataset/
                human_test_easy.json
                human_test_hard.json
                test_easy.json
                test_hard.json
    .archived/
        w4_fuzzydistance/
            consts.py
            embed_processing.py
            embed_reporting.py
            main.py
            strip_questions.py
            strip_range_data.py
"""

from strip_questions import strip_questions
from strip_range_data import strip_ranges
from pathlib import Path

# easy only by default, regex not updated for hard
USE_EASY = True
USE_HARD = False
# disable to skip embedding (model load is slow)
EXTRACT_EMBEDDINGS = True


def _process_timeqa_question_files():
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "timeqa" / "dataset"
    test_paths = {
        "test_easy": dataset_path / "test_easy.json",
        "test_hard": dataset_path / "test_hard.json",
        "human_test_easy": dataset_path / "human_test_easy.json",
        "human_test_hard": dataset_path / "human_test_hard.json",
    }

    if USE_EASY:
        print("Processing easy questions...")
        # check if exists
        if not test_paths["human_test_easy"].exists():
            print(f"File not found: {test_paths['human_test_easy']}, toggle main.USE_EASY to False if skipping, check file exists and/or update the correct path.")
        else:
            strip_questions(test_paths["human_test_easy"])
        
        if not test_paths["test_easy"].exists():
            print(f"File not found: {test_paths['test_easy']}, toggle main.USE_EASY to False if skipping, check file exists and/or update the correct path.")
        else:
            strip_questions(test_paths["test_easy"])
    
    if USE_HARD:
        print("Processing hard questions...")
        # check if exists
        if not test_paths["human_test_hard"].exists():
            print(f"File not found: {test_paths['human_test_hard']}, toggle main.USE_HARD to False if skipping, check file exists and/or update the correct path.")
        else:
            strip_questions(test_paths["human_test_hard"])
        
        if not test_paths["test_hard"].exists():
            print(f"File not found: {test_paths['test_hard']}, toggle main.USE_HARD to False if skipping, check file exists and/or update the correct path.")
        else:
            strip_questions(test_paths["test_hard"])


def main():
    # step 1: strip questions from dataset
    _process_timeqa_question_files()

    # step 2: build fuzzy ranges from extracted ranges
    # (no args needed)
    print("Processing extracted ranges to authorative data file...")
    strip_ranges()

    if EXTRACT_EMBEDDINGS:
        from embed_processing import main as embed_main
        from embed_reporting import main as embed_report
        # steps 3 & 4 optional
        #TODO split reporting from embedding so the report can rerun without re-embedding (use authorative data file as input)
        # step 3: embed range data
        embed_main()
        # step 4: compare and plot
        embed_report()


#TODO add cli args
if __name__ == "__main__":
    main()

