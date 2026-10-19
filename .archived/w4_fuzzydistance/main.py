"""
This weeks experiment is comparing embedding distance between explicit date ranges and fuzzy date ranges, using TimeQA questions as the source data.
w4_fuzzydistance intended as a self-contained module - this is an access function to call the other scripts in this module

(This script assumes that the other scripts are in the same directory as this file)
Expected directory structures:
    ./data/
        timeqa/
            dataset/
                human_test_easy.json
                human_test_hard.json
                test_easy.json
                test_hard.json
    ./src/
        w4_fuzzydistance/
            embed_range_data.py
            strip_questions.py
            strip_range_data.py
            consts.py
            main.py
"""

from embed_processing import main as embed_main
from embed_reporting import main as embed_report
from strip_questions import strip_questions
from strip_range_data import strip_ranges
from pathlib import Path

# defaults to using easy questions as regex not updated for hard
USE_EASY = True
USE_HARD = False
# model loading takes time, disable this if you want to just run the question stripping and range extraction without embedding
EXTRACT_EMBEDDINGS = True


def _process_timeqa_question_files():
    test_paths = {
        "test_easy": Path("data/timeqa/dataset/test_easy.json"),
        "test_hard": Path("data/timeqa/dataset/test_hard.json"),
        "human_test_easy": Path("data/timeqa/dataset/human_test_easy.json"),
        "human_test_hard": Path("data/timeqa/dataset/human_test_hard.json"),
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
    # Step 1: Strip questions from relevant dataset
    _process_timeqa_question_files()

    # Step 2: Process the extracted ranges to create fuzzy categories
    # (no arguments required, function self-contained)
    print("Processing extracted ranges to authorative data file...")
    strip_ranges()

    if EXTRACT_EMBEDDINGS:
        # Step 3 & 4 are optional
        #TODO - separate reporting from embedding to allow for re-running the report without re-embedding (use authorative data file as input)
        # Embed the range data
        embed_main()
        # Compare and visualise
        embed_report()


#TODO add CLI args
if __name__ == "__main__":
    main()

