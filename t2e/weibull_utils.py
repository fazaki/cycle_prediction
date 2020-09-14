import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


def weibull_pdf(alpha, beta, t):
    return (beta/alpha) *\
        ((t+1e-35)/alpha)**(beta-1)*np.exp(- ((t+1e-35)/alpha)**beta)


def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)


def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)


def weibull_mode(alpha, beta):
    if beta < 1:
        return 0
    return alpha * ((beta-1)/beta)**(1/beta)

# def plot_top_predictions(
    # result_df,
    # lim=10,
    # top_feature = "abs_error",
    # ascending = True,
    # U = 1,
    # accurate = True
    # ):

#     result_df = result_df.loc[(result_df["Accurate"] == accurate) &\
#       (result_df["U"] == U)]
#     top_accurate = result_df.sort_values(by=top_feature,
#                                           ascending = ascending)
#                                          .head(lim).index
#     result_df.sort_values(by=top_feature, ascending =\
#       ascending).head(lim)
#     fig, axarr = plt.subplots(len(top_accurate),
#       figsize=(15,len(top_accurate)*3))

#     for n,i in enumerate(top_accurate):
#         ax = axarr[n]
#         T    = result_df.loc[i,'T']
#         U    = result_df.loc[i,'U']
#         alpha= result_df.loc[i,'alpha']
#         beta = result_df.loc[i,'beta']
#         mode = result_df.loc[i,'T_pred']

#         y_max_1 = weibull_pdf(alpha, beta, mode)
#         y_max_2 = weibull_pdf(alpha, beta, T)

#         t=np.arange(0,20)
#         ax.plot(t, weibull_pdf(alpha, beta, t), color='w',
#           label="Weibull distribution")
#         ax.vlines(mode, ymin=0, ymax=y_max_1, colors='k',
#           linestyles='--', label="Predicted failure time")
#         ax.scatter(T, weibull_pdf(alpha,beta, T), color='r',
#           s=100, label="Actual failure time")
#         ax.set_facecolor('#A0A0A0')
#         ax.xaxis.grid(b=True, which="major", color='k',
#           linestyle='-.', linewidth=0.1)
#         ax.set_xticklabels(np.append(0, np.unique(t)))
#         ax.set_xlim(left = 0)
#         ax.set_ylim(bottom = 0)
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

#         if n == 0:
#             ax.legend(frameon=True,fancybox=True,shadow=True,facecolor='lightgray')
#             ax.set_title("Time-to-Failure distribution",pad =20,
#               fontsize = 20)
#     #plt.tight_layout()
#     plt.show()


def plot_predictions_insights(results_df):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    observed = results_df.loc[results_df['U'] == 1]
    sns.scatterplot(observed['T'], observed["T_pred"],
                    hue=observed["Accurate"])
    plt.xlabel("Actual failure time", fontsize=12)
    plt.ylabel("Predicted failure time", fontsize=12)
    plt.title('Actual Vs. Predicted (Observed)', fontsize=18)
    plt.legend(loc='upper left')
    plt.subplot(2, 2, 2)
    censored = results_df.loc[results_df['U'] == 0]
    sns.scatterplot(censored['T'], censored["T_pred"],
                    hue=censored["Accurate"])
    plt.xlabel("Time of observation", fontsize=12)
    plt.ylabel("Predicted failure time", fontsize=12)
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
