def easy_grader(action, observation, info):
    actual = info.get("actual")
    return 1.0 if action == actual else 0.0


def medium_grader(action, observation, info):
    actual = info.get("actual")
    if action == actual:
        return 1.0
    return 0.5   # partial credit


def hard_grader(action, observation, info):
    actual = info.get("actual")
    return 1.0 if action == actual else 0.0