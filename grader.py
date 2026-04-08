def easy_grader(prediction, actual):
    return 1.0 if prediction == actual else 0.0


def medium_grader(prediction, actual):
    if prediction == actual:
        return 1.0
    return 0.5  # partial reward


def hard_grader(prediction, actual):
    if prediction == actual:
        return 1.0
    return 0.0