def grade(task_id, prediction, ground_truth):

    # correct prediction
    if prediction == ground_truth:
        return 1.0

    # partial credit for hard tasks
    if task_id == "hard":
        return 0.3

    return 0.0