import numpy as np
def soca_cfar(half_train, half_guard, SOS, data):

    ns = len(data) # number of samples
    result = np.zeros(ns)
    th = np.zeros(ns)
    lead = half_train + half_guard # max num cells considered on either side of cut
    lag = ns - lead
    for cutidx in range(lead,lag): #cutidx = index of cell under test
        
        # extract training sets
        lhs_train = data[cutidx-lead:cutidx-half_guard]
        rhs_train = data[cutidx+half_guard:cutidx+lead]

        cut = data[cutidx]
        ZOS = min(np.average(lhs_train),np.average(rhs_train))
        TOS = SOS*ZOS
        th[cutidx] = TOS
        # print('TOS =', th[cutidx])
        if cut > TOS:
            # index implies frequency. return magnitude for use in
            # determining max value
            result[cutidx] = cut

    return result, th