def easy_grader(action, observation, info):
    actual = info.get("actual")
    return 0.95 if action == actual else 0.05


def medium_grader(action, observation, info):
    actual = info.get("actual")
    return 0.95 if action == actual else 0.35


def hard_grader(action, observation, info):
    actual = info.get("actual")
    return 0.95 if action == actual else 0.05