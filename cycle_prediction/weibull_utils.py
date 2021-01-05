import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


def weibull_pdf(alpha, beta, t):
    return (beta/alpha) *\
        ((t+1e-35)/alpha)**(beta-1)*np.exp(- ((t+1e-35)/alpha)**beta)


def weibull_survival(alpha, beta, t):
    return np.exp(-((t+1e-35) / alpha) ** beta)


def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)


def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)


def weibull_mode(alpha, beta):
    if beta < 1:
        return 0
    return alpha * ((beta-1)/beta)**(1/beta)


def plot_top_predictions(
        result_df,
        lim=10,
        U=1,
):
    """Plot weibull pdf and survival curve for then remaining time.

    Args:
        result_df: the result dataframe from the t2e.evaluate method
        lim: limit to plot the best n predictions
        U: 1:observed or 0:censored example to plot
    """
    result_df = result_df.loc[result_df['Accurate'] == True]
    top_features = ['T_pred', 'MAE']
    sorting_bool = [False, True]
    result_df = result_df.loc[result_df["U"] == U]
    result_df_noZero = result_df.loc[result_df["T"] != 0]
    result_df_noZero = result_df_noZero.loc[result_df["T_pred"] != 0]
    if len(result_df_noZero) != 0:
        result_df = result_df_noZero
    result_df = result_df.sort_values(by=top_features,
                                      ascending=sorting_bool)\
                         .reset_index(drop=True)\
                         .head(lim)
    fig, axarr = plt.subplots(nrows=lim, ncols=2, figsize=(20, lim*4))

    max_time = max(
        result_df.loc[:, 'T'].max(),
        result_df.loc[:, 'T_pred'].max()
        )
    if max_time == 0:
        max_time = 15
    elif max_time <= 15:
        max_time = max_time*10
    else:
        max_time = max_time*5
    for i in range(lim):
        try:
            ax = axarr[i][0]
        except Exception:
            ax = axarr[0]
        T = result_df.loc[i, 'T']

        alpha = result_df.loc[i, 'alpha']
        beta = result_df.loc[i, 'beta']
        mode = result_df.loc[i, 'T_pred']

        y_max_1 = weibull_pdf(alpha, beta, mode)
        # y_max_2 = weibull_pdf(alpha, beta, T)
        t = np.arange(0, max_time)
        ax.plot(t, weibull_pdf(alpha, beta, t), color='gray',
                label="Weibull distribution")
        ax.vlines(mode, ymin=0, ymax=y_max_1, colors='k',
                  linestyles='--', label="Pred. remaining time")
        ax.scatter(T, weibull_pdf(alpha, beta, T), color='g',
                   s=100, label="Actual remaining time")
        ax.set_facecolor('lightgray')
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-1, right=max_time)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True,
                      facecolor='w', fontsize=18)
            ax.set_title("Remaining time distribution", pad=20,
                         fontsize=20)
            ax.set_xlabel("Remaining time", fontsize=15)
            ax.set_ylabel("f(t)", fontsize=15, rotation=360, labelpad=18)
        ###################
        #  Survival plot  #
        ###################
        try:
            ax = axarr[i][1]
        except Exception:
            ax = axarr[1]
        t = np.arange(0, max_time)
        ax.plot(t, weibull_survival(alpha, beta, t), color='gray',
                label="Weibull survival curve")
        ax.set_facecolor('lightgray')
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-1, right=max_time)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True,
                      facecolor='w', fontsize=18)
            ax.set_title("Process survival curve", pad=20,
                         fontsize=20)
            ax.set_xlabel("Remaining time", fontsize=15)
            ax.set_ylabel("S(t)", fontsize=15, rotation=360, labelpad=18)
    plt.tight_layout()
    plt.show()


def plot_predictions_insights(results_df):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    observed = results_df.loc[results_df['U'] == 1]
    sns.scatterplot(observed['T'], observed["T_pred"],
                    hue=observed["Accurate"])
    plt.xlabel("Actual remaining time", fontsize=12)
    plt.ylabel("Predicted remaining time", fontsize=12)
    plt.title('Actual Vs. Predicted (Observed)', fontsize=18)
    plt.legend(loc='upper left')
    plt.subplot(2, 2, 2)
    censored = results_df.loc[results_df['U'] == 0]
    sns.scatterplot(censored['T'], censored["T_pred"],
                    hue=censored["Accurate"])
    plt.xlabel("Time of observation", fontsize=12)
    plt.ylabel("Predicted remaining time", fontsize=12)
    plt.title('Actual Vs. Predicted (Censored)', fontsize=18)
    plt.legend(loc='upper left')
    plt.subplot(2, 2, (3, 4))
    sns.distplot(results_df['error (days)'], bins=100,
                 kde=True, hist=True, norm_hist=False,
                 label="Error Rate")
    plt.title('Error distribution', fontsize=18)
    plt.legend(loc='upper left')
    plt.xlabel("Error Shift", fontsize=12)
#     plt.subplot(3,2,(5,6))
#     x = results_df.groupby(["T","U"]).agg({"Accurate":"mean"}).reset_index()
#     x["Error_Rate"] = 1-x["Accurate"]
#     sns.barplot(x= x["T"].astype("int"), y=x["Error_Rate"], hue=x["U"])
#     plt.title('Error rate per time step')
    plt.tight_layout()
    plt.show()


def check_dir(dir_path, mkdir=True):
    dir_exist = os.path.exists(dir_path)
    if not dir_exist and mkdir:
        os.makedirs(dir_path)
    tmp_dir_exist = os.path.exists(dir_path)
    if not tmp_dir_exist and mkdir:
        raise ValueError("Failed to create dir '%s'" % dir_path)
    return dir_exist
