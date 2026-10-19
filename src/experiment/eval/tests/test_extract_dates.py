from experiment.preprocessing.extract_dates import extract_paragraph_date_span


def test_population_count_not_a_year():
    text = "The settlements had populations of at least 200 people (populations as of 2020)."
    span = extract_paragraph_date_span(text)
    assert span[0] != 200
    assert span == (2020, 2020)


def test_page_reference_not_a_year():
    text = "( S. Fischer : 2013 ) ... Blackwell Companions to World History"
    date_start, _ = extract_paragraph_date_span(text)
    assert date_start != 432


def test_acreage_not_a_year():
    text = "Another 368 acres south of Odder has been protected since 1979"
    span = extract_paragraph_date_span(text)
    assert span[0] != 368
    assert span == (1979, 1979)


def test_constituency_number_not_a_year():
    text = "...contested the marginal Norwich North constituency at the 1992 general election"
    date_start, _ = extract_paragraph_date_span(text)
    assert date_start != 0


def test_month_year():
    assert extract_paragraph_date_span("May 1997") == (1997, 1997)


def test_day_month_year():
    assert extract_paragraph_date_span("12 May 1997") == (1997, 1997)


def test_year_range():
    assert extract_paragraph_date_span("1997 to 2001") == (1997, 2001)


def test_no_date_found():
    assert extract_paragraph_date_span("No date mention here at all.") == (None, None)
