from app.evaluation import _itemset_metrics


def test_itemset_metrics_perfect_recovery():
    reference = {frozenset({"CS101", "CS102"}), frozenset({"AI201"})}
    predicted = set(reference)

    precision, recall, f1 = _itemset_metrics(reference, predicted)

    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_itemset_metrics_partial_recovery():
    reference = {frozenset({"CS101", "CS102"}), frozenset({"AI201"})}
    predicted = {frozenset({"CS101", "CS102"}), frozenset({"WD201"})}

    precision, recall, f1 = _itemset_metrics(reference, predicted)

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_itemset_metrics_empty_sets():
    assert _itemset_metrics(set(), set()) == (0.0, 0.0, 0.0)
