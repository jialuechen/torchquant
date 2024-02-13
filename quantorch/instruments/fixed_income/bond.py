class Bond():
    def __init__(self) -> None:
        pass

    def bond_price_discrete(times,cashflows,r):
        p=0
        for i in range(len(times)):
            p+=cashflows[i]/np.power((1+r),times[i])
        return p