from statsmodels.multivariate.manova import MANOVA
import scipy.stats as stats


def do_manova_on_groups(vectors, labels):
    manova = MANOVA(endog=vectors, exog=labels)
    print(manova.mv_test())


def do_anova_on_groups(values, labels):
    groups = [[] for i in range((max(labels) + 1))]
    for v, l in zip(values, labels):
        groups[l].append(v)

    print("ANOVA: ")

    if len(groups) == 5:
        print(stats.f_oneway(groups[0], groups[1], groups[2], groups[3], groups[4]))
    elif len(groups) == 4:
        print(stats.f_oneway(groups[0], groups[1], groups[2], groups[3]))
    elif len(groups) == 3:
        print(stats.f_oneway(groups[0], groups[1], groups[2]))
    elif len(groups) == 2:
        print(stats.f_oneway(groups[0], groups[1]))


def do_kruskal_wallis_test(values, labels):
    groups = [[] for i in range((max(labels) + 1))]
    for v, l in zip(values, labels):
        groups[l].append(v)

    print("Kruskal: ")

    if len(groups) == 5:
        print(stats.kruskal(groups[0], groups[1], groups[2], groups[3], groups[4]))
    elif len(groups) == 4:
        print(stats.kruskal(groups[0], groups[1], groups[2], groups[3]))
    elif len(groups) == 3:
        print(stats.kruskal(groups[0], groups[1], groups[2]))
    elif len(groups) == 2:
        print(stats.kruskal(groups[0], groups[1]))






