
# Early Stopping - But When? Lutz Prechelt
# If stopping function return True , mean it's time to stopping train,else continue.

def gl(best_vail_error,vail_error):
    """
    GL(t) = 100 * (best_vail_error/vail_error -1)
    :param best_vail_error: The value Eopt(t) is defined to be the lowest validation set error obtained in epochs up to t
    :param vail_error: as your knows
    :return: GL(t)
    """
    gl = 100 * (best_vail_error/vail_error -1)


def default_early_stopping(train_error,vail_error):

    if vail_error[-1]>vail_error[-2]:

        return True
    else:

        return False
