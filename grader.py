def easy_grader(action, observation, info):
    actual = info.get("actual")
    return 1.0 if action == actual else 0.0


def medium_grader(action, observation, info):
    actual = info.get("actual")
    return 1.0 if action == actual else 0.5


def hard_grader(action, observation, info):
    actual = info.get("actual")
    return 1.0 if action == actual else 0.0