def drug_user(
    prob_th=0.8,
    sensitivity=0.79,
    specificity=0.79,
    prevelence=0.02,
    verbose=True):
    p_user=prevelence
    p_non_user=1-prevelence
    p_pos_user=sensitivity
    p_neg_user=specificity
    p_pos_non_user=1-specificity
    num=p_pos_user*p_user
    den=p_pos_user*p_user+p_pos_non_user*p_non_user
    prob=num/den
    if verbose:
        if prob>prob_th:
            print("The test taker could be a user")
        else:
            print("The test taker may not be a user")
    return prob
print("Ashirvaad")
p=drug_user(prob_th=0.5,sensitivity=0.97,specificity=0.95,prevelence=0.005)
print("Probability of the test taker being a drug use is:",round(p,3)) 