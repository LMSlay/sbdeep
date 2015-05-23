
# ref: Early Stopping - But When? Lutz Prechelt
# If stopping function return True , mean it's time to stopping train,else continue.

def gl(best_vail_error,vail_error):
    """
    GL(t) = 100 * (vail_error/best_vail_error -1)
    :param best_vail_error: The value Eopt(t) is defined to be the lowest validation set error obtained in epochs up to t
    :param vail_error: as your knows
    :return: GL(t)
    """
    gl = 100 * (vail_error/best_vail_error -1.)
    return gl

def gls(best_vail_error,vail_error):
    """
    GL(t) = 100 * (vail_error/best_vail_error -1)
    :param best_vail_error: The value Eopt(t) is defined to be the lowest validation set error obtained in epochs up to t
    :param vail_error: as your knows
    :return: GL(t)
    """
    gl = 100 * (best_vail_error/vail_error -1.)
    return gl


def default_early_stopping(train_error,vail_error):

    if len(vail_error)<2:
        return False


    if vail_error[-1]>vail_error[-2]:

        return True
    else:

        return False


def st1_early_stopping(train_error, vail_error, alphe=20.):

    expr = gl(best_vail_error=min(vail_error),vail_error=vail_error[-1])

    if expr>alphe:
        return True
    else:
        return False


def st2_early_stopping(train_error, vail_error, alphe=500, strips=5):
    # shit!!!!
    if len(train_error)<strips:
        return False

    train_av = 0.
    for i in train_error[-strips:]:
        train_av += i
    pk = 1000 * ((strips*min(train_error))/train_av-1)

    expr =  pk/gls(best_vail_error=min(vail_error), vail_error=vail_error[-1])

    if expr>alphe:
        return True
    else:
        return False

def st3_early_stopping(train_error, vail_error, strips=10):

    if len(vail_error)<strips:
        return False

    inc = vail_error[-strips]
    for i,e in enumerate(vail_error[-strips:]):

        if e<inc:
            return False

        if e>inc:
            inc = e

        if i==strips-1:
            return True

