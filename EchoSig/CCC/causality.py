from __future__ import annotations
import matplotlib.pyplot as plt
import os
from scipy.stats import false_discovery_control


# from statsmodels.compat.numpy import lstsq
# from statsmodels.compat.pandas import deprecate_kwarg
# from statsmodels.compat.python import lzip
# from statsmodels.compat.scipy import _next_regular

# from typing import Literal, Union
import warnings
# from pandas import DataFrame
# from statsmodels.tools.typing import NDArray
import numpy as np
# from numpy.linalg import LinAlgError
# import pandas as pd
from scipy import stats
# from scipy.interpolate import interp1d
from scipy.signal import correlate

from statsmodels.regression.linear_model import OLS #, yule_walker
from statsmodels.tools.sm_exceptions import (
    # CollinearityWarning,
    InfeasibleTestError,
    # InterpolationWarning,
    # MissingDataError,
    # ValueWarning,
)
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    # dict_like,
    # float_like,
    int_like,
    # string_like,
)
# from statsmodels.tsa._bds import bds
# from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
# from statsmodels.tools.data import _is_recarray, _is_using_pandas
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr,pearsonr,kendalltau
from sklearn.metrics import mean_squared_error
from dtaidistance import dtw
import pandas as pd
warnings.filterwarnings("ignore")



def spearman_score(y_true,y_pred):
    return spearmanr(y_true,y_pred)[0]
def pearson_score(y_true,y_pred):
    return pearsonr(y_true,y_pred)[0]
def kendalltau_score(y_true,y_pred):
    return kendalltau(y_true,y_pred)[0]
METRICS = {
    'rmse':mean_squared_error,
    'dtw':dtw.distance,
    'spearman':spearman_score,
    'pearson':pearson_score,
    'kendalltau':kendalltau_score,
    }
def CCC_grangercausalitytests(x, maxlag,causal_stc=['spg',['ligand','receptor']], addconst=True, verbose=None):
    """
    CCC granger causality test for 3 time series. 

    All four tests give similar results. `params_ftest` and `ssr_ftest` are
    equivalent based on F test which is identical to lmtest:grangertest in R.

    Parameters
    ----------
    x : pd.DataFrame
        A set of time series data to examine their causality relationship
    causal_stc: list
        The causal structure to determine their time granger-like causality. 
        Each element is a str that is the key for `x`.
        1. It can be a list of two str. e.g., ['a','b']. It examine if x['b'] granger causes x['a']
              -- This is equivalent to granger causality
        2. It can contain three variables. e.g., ['spg',['ligand','receptor']]. 
           It examines if the pair in second element granger causes the first element.
               i.e., does ligand-recpetor pair: L(t-lag)*R(t) causes SPG(t)
               The time lag only consider for ligand.
                -- This is a modified granger causality for Ligand-Receptor-SPG triplets.
        
        e.g., 
    maxlag : {int, Iterable[int]}
        If an integer, computes the test for all lags up to maxlag. If an
        iterable, computes the tests only for the lags in maxlag.
    addconst : bool
        Include a constant in the model.
    verbose : bool
        Print results. Deprecated

        .. deprecated: 0.14

           verbose is deprecated and will be removed after 0.15 is released



    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        test statistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    `params_ftest`, `ssr_ftest` are based on F distribution

    `ssr_chi2test`, `lrtest` are based on chi-square distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Granger_causality

    .. [2] Greene: Econometric Analysis

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.stattools import grangercausalitytests
    >>> import numpy as np
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> data = data.data[["realgdp", "realcons"]].pct_change().dropna()

    All lags up to 4

    >>> gc_res = grangercausalitytests(data, 4)

    Only lag 4

    >>> gc_res = grangercausalitytests(data, [4])
    """
    assert len(causal_stc)==2
    if isinstance(causal_stc[0],str) and isinstance(causal_stc[1],str):
        x = array_like(x[causal_stc],"x",ndim=2)
        conv_granger=True
    elif isinstance(causal_stc[0],str) and isinstance(causal_stc[1],list):
        assert len(causal_stc[1])==2
        x = array_like(x[[
            causal_stc[0],
            causal_stc[1][0],
            causal_stc[1][1]]
            ],
            "x",
            ndim=2)
        conv_granger=False
    # x = array_like(x, "x", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values.")
    addconst = bool_like(addconst, "addconst")
    if verbose is not None:
        verbose = bool_like(verbose, "verbose")
        warnings.warn(
            "verbose is deprecated since functions should not print results",
            FutureWarning,
        )
    else:
        verbose = True  # old default

    try:
        maxlag = int_like(maxlag, "maxlag")
        if maxlag <= 0:
            raise ValueError("maxlag must be a positive integer")
        lags = np.arange(1, maxlag + 1)
    except TypeError:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError(
                "maxlag must be a non-empty list containing only "
                "positive integers"
            )

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {}".format(int((x.shape[0] - int(addconst)) / 3) - 1)
        )

    resli = {}

    for mlg in lags:
        result = {}
        if verbose:
            print("\nGranger Causality")
            print("number of lags (no zero)", mlg)
        mxlg = mlg
        if conv_granger:
            # create lagmat of both time series
            dta = lagmat2ds(x, mxlg, trim="both", dropex=1)
        else:
            dta_tmp = lagmat2ds(x[:,0:2], mxlg, trim="both", dropex=1)
            receptor_current = x[mxlg:,2] 
            dta = dta_tmp
            dta[:,mxlg+1:] = np.sqrt(dta[:,mxlg+1:] * receptor_current.reshape(-1,1))

        # add constant
        if addconst:
            dtaown = add_constant(dta[:, 1 : (mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            if (
                dtajoint.shape[1] == (dta.shape[1] - 1)
                or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
            ):
                raise InfeasibleTestError(
                    "The x values include a column with constant values and so"
                    " the test statistic cannot be computed."
                )
        else:
            raise NotImplementedError("Not Implemented")
            # dtaown = dta[:, 1:mxlg]
            # dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # print results
        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be computed "
                "because the VAR has a perfect fit of the data."
            )
        fgc1 = (
            (res2down.ssr - res2djoint.ssr)
            / res2djoint.ssr
            / mxlg
            * res2djoint.df_resid
        )
        if verbose:
            print(
                "ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (
                    fgc1,
                    stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                    res2djoint.df_resid,
                    mxlg,
                )
            )
        result["ssr_ftest"] = (
            fgc1,
            stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
            res2djoint.df_resid,
            mxlg,
        )

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print(
                "ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, "
                "df=%d" % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)
            )
        result["ssr_chi2test"] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print(
                "likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d"
                % (lr, stats.chi2.sf(lr, mxlg), mxlg)
            )
        result["lrtest"] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack(
            (np.zeros((mxlg, mxlg)), np.eye(mxlg, mxlg), np.zeros((mxlg, 1)))
        )
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print(
                "parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)
            )
        result["params_ftest"] = (
            np.squeeze(ftres.fvalue)[()],
            np.squeeze(ftres.pvalue)[()],
            ftres.df_denom,
            ftres.df_num,
        )

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli

def compute_lagged_values(target,ligand,lag_list,receptor=None,metrics = mean_squared_error):
    """ Compute stat of signal with different lag values

    Args:
        target (np.array): A 1D NumPy array representing a time series for target gene in receiver cell.
        ligand (np.array): A 1D NumPy array representing a time series for ligand in sender cell.
        lag_list (np.array or list): An array or list of indices for candidate lag steps for evaluation.
        receptor (np.array, optional): A 1D NumPy array representing a time series for receptor in receiver cell.
            If it is None, we consider the signal from ligand to target without receptor
            Defaults to None.
        metrics (Callable, optional): A function that computes a metrics between two time series. 
            Defaults to mean_squared_error.

    Returns:
        stat (np.array): A 1D NumPy array representing statistical values of the L-R-SPG signal under different lags.
    """
    if receptor is None:
        receptor = np.ones_like(ligand)
    assert not (np.any(np.isnan(ligand)) or np.any(np.isnan(receptor)) or np.any(np.isnan(target))), 'found NAN in input time series'
    # p=[]
    stat=[]
    for lag in lag_list:
        if lag==0:
            a = metrics(np.sqrt(ligand*receptor),target)
        elif lag>0:
            a = metrics(np.sqrt(ligand[:-lag]*receptor[lag:]),target[lag:])
        else: 
            a = metrics(np.sqrt(ligand[-lag:]*receptor[:lag]),target[:lag])
        # p.append(a[1])
        stat.append(a)
    # p=np.array(p)
    stat = np.array(stat)
    # idx = np.argmin(p)
    # idx=np.nanargmax(stat)
    # # opt_p = np.min(p)
    # opt_stat=stat[idx]
    # opt_lag=lag_list[idx]*dt
    return stat

def find_lag_regulate(target,ligand,lag_list,receptor=None,metrics='rmse',curve_thred=0.2,p_thred=0.05):
    """ Find the optimal time lag and the regulatory effect (activation or inhibition).

    Args:
        target (np.array): A 1D NumPy array representing a time series for target gene in receiver cell.
        ligand (np.array): A 1D NumPy array representing a time series for ligand in sender cell.
        lag_list (np.array or list): An array or list of indices for candidate lag steps for evaluation.
        receptor (np.array, optional): A 1D NumPy array representing a time series for receptor in receiver cell.
            If it is None, we consider the signal from ligand to target without receptor
            Defaults to None.
        receptor (np.array, optional): A 1D NumPy array representing a time series for receptor in receiver cell.
            If it is None, we consider the signal from ligand to target without receptor
            Defaults to None.        
        metrics (str, optional): string name of the metrics. See keys in `METRICS` for values. 
            Defaults to 'rmse'.

    Returns:
        sign (-1,0,1): Regulatory effect. `1` is activation, `-1` is inhibition, and `0` is no statistically signficiant regulatory effect
        opt_lag (float): lag value for the regulatory effect. It is `-1`, if there is no no statistically signficiant regulatory effect.
        opt_stat (float): the score for the regulatory effect. It is `np.nan`, if there is no no statistically signficiant regulatory effect.
        p_value (float): the statistical p-value of the regulatory effect
    """
                

    if (np.max(ligand)-np.min(ligand))<curve_thred \
        or (np.max(target)-np.min(target))<curve_thred:
        opt_lag=np.nan
        opt_stat = np.nan
        sign=0
        p_value=1.
        stat=[]
        return sign, opt_lag, opt_stat,p_value,stat
    if metrics in ['rmse','dtw']:
        pos_stat=compute_lagged_values(target,ligand,lag_list,receptor, METRICS.get(metrics))    
        neg_stat=compute_lagged_values(np.max(target)+np.min(target)-target,ligand,lag_list,receptor, METRICS.get(metrics))    
        
        opt_pos_stat = np.min(pos_stat)
        opt_neg_stat = np.min(neg_stat)
        # plt.plot(pos_stat)
        # plt.plot(neg_stat)
        # plt.legend(['activation test','inhibition test'])
        # stat=np.append(pos_stat,neg_stat)
        pos_thred = np.percentile(pos_stat,75)
        top_pos_stat = pos_stat[pos_stat>pos_thred]
        neg_thred = np.percentile(neg_stat,75)
        top_neg_stat = neg_stat[neg_stat>neg_thred]
        _,p_value = mannwhitneyu(top_pos_stat,top_neg_stat,alternative='two-sided')
        if p_value<p_thred:
            # there could be a time lag for causal relationship
            if opt_pos_stat>=opt_neg_stat:
                stat = neg_stat
                sign=-1 
            else:
                stat = pos_stat
                sign=1
            idx = np.argmin(stat)
            opt_stat = stat[idx]
            opt_lag = lag_list[idx]

            if opt_lag<=0:
                opt_lag=np.nan
                opt_stat = np.nan
                sign=0
        else:
            sign=0
            opt_lag=np.nan
            opt_stat = np.nan
            stat=np.append(pos_stat,neg_stat)
        return sign, opt_lag, opt_stat,p_value,stat


def evaluate_causality(target,ligand,lag_list,max_lag,dt,receptor=None,metrics='rmse',
                       max_target=1.,max_ligand=1.,max_receptor=1.,curve_thred=0.2,p_thred=0.05,
                       time_unit='h'):
    """_summary_

    Args:
        target (np.array): A 1D NumPy array representing a time series for target gene in receiver cell.
        ligand (np.array): A 1D NumPy array representing a time series for ligand in sender cell.
        lag_list (np.array or list): An array or list of indices for candidate lag steps for evaluation.
        max_lag (float): Maximal allowance of lag step. 
            If the inferred lag step exceed `max_lag`* N (length of trajectory), no causality.
        dt (float): Time step, representing the interval between successive time points.
        receptor (np.array, optional): A 1D NumPy array representing a time series for receptor in receiver cell.
            If it is None, we consider the signal from ligand to target without receptor
            Defaults to None.
        receptor (np.array, optional): A 1D NumPy array representing a time series for receptor in receiver cell.
            If it is None, we consider the signal from ligand to target without receptor
            Defaults to None.        
        metrics (str, optional): string name of the metrics. See keys in `METRICS` for values. 
            Defaults to 'rmse'.
        max_target (float, optional): global max of target
            We recommand using the global max from the whole dataset than the single trajectory
            Defaults to 1..
        max_ligand (float, optional): global max of ligand
            We recommand using the global max from the whole dataset than the single trajectory
            Defaults to 1..
        max_receptor (float, optional): global max of receptor
            We recommand using the global max from the whole dataset than the single trajectory
            Defaults to 1..
        curve_thred (float, optional): thredhold to filter out curves with low change magnitude.
            Default to 0.2

    Returns:
        _type_: _description_
    """
    # p,stat,opt_p,opt_stat,opt_lag = lag_search(target,ligand,lag_list, dt,metrics=metrics)
    ligand = ligand/max_ligand
    target = target/max_target     

    if receptor is None:
        # with_receptor = False
        receptor = np.ones_like(ligand)
        ligand=ligand**2 # make sure signal and target having the consistent unit
    else: 
        # with_receptor = True
        receptor = receptor/max_receptor
    # max normalization of the data. Preserve the variance of coefficient.
    # And allow all curves having the same max values.
   
        
    sign, opt_lag, opt_stat,p_value, stat = find_lag_regulate(target,ligand,lag_list,receptor=receptor, metrics=metrics,
                                                              curve_thred=curve_thred,p_thred=p_thred)
        # sign_lr, opt_lag_lr, _,_, _ = find_lag_regulate(target,ligand,lag_list,receptor=None, metrics=metrics)
    # else:
        # sign, opt_lag, opt_stat,p_value, stat = find_lag_regulate(target,ligand,lag_list,receptor=receptor, metrics=metrics)
    

    if sign<0:
        target = np.max(target)+np.min(target)-target
    if opt_lag>0 and max_lag*target.shape[0]>opt_lag:
    # opt_lag<= int((target.shape[0] - 1) / 3) - 1:
        # if receptor is None:
        #     receptor = np.ones_like(ligand)
        granger = CCC_grangercausalitytests(pd.DataFrame({'target':target,
                                                      'ligand':ligand,
                                                      'receptor':receptor}), 
                                                      [opt_lag],
                                                      ['target',['ligand','receptor']], 
                                                      verbose=False)
        
        ftest_p = granger[opt_lag][0]['ssr_ftest'][1]
        ftest_stat = granger[opt_lag][0]['ssr_ftest'][0]
        chi2_p = granger[opt_lag][0]['ssr_chi2test'][1]
        chi2_stat = np.log1p(granger[opt_lag][0]['ssr_chi2test'][0])
    else:
        ftest_p = 1.
        ftest_stat = 0.
        chi2_p = 1.
        chi2_stat = 0.

    data = {'lag':opt_lag*dt,
            metrics: opt_stat,
            'lag p-value':p_value,
            'p ftest' : ftest_p,
            'stat ftest' : ftest_stat*sign,
            'p chi2' : chi2_p,
            'stat chi2' :chi2_stat*sign, 
            metrics + ' verbose': stat,    
            # 'lag verbose': lag_list,
            'sign':sign,
            'time_unit':time_unit,
            # 'sign_lr':sign_lr,
            # 'confound': True if sign*sign_lr == -1 else False,

            }
    # if with_receptor:
    #     data['sign_lr'] = sign_lr
    #     data['lag_lr'] = opt_lag_lr*dt
    #     data['confound'] = True if sign*sign_lr == -1 else False

    # data = {'lag':opt_lag*dt,
    #         'p':ftest_p,
    #         'stat': sign*ftest_stat,
    #         }
    return data




def trajCCC(source_id,target_id,z_traj,time,lag_list,max_lag,
            df_LRSPG,gene_list,
            save_dir,
            curve_thred=0.2, p_thred=0.01,save_fig=False,time_unit=None,time_scale=1):
    """ Calculate CCC causality between two cell trajectories for given L-R-SPG info

    Args:
        source_id (int): index of source cell
        target_id (int): index of target cell
        z_traj (np.array): 
            Represents the gene expression trajectories over time for each cell.
            They need to be normalized by the global max.
            A 3D array of shape (N, T, G):
                N is the number of cells
                T is number of time
                G is number of genes
        time (np.array): A 1D array of shape (T,) for T time points if the trajectory.
        lag_list (np.array or list): An array or list of indices for candidate lag steps for evaluation.
        max_lag (float): Maximal allowance of lag step. 
            If the inferred lag step exceed `max_lag`* N (length of trajectory), no causality.
        df_LRSPG (pd.DataFrame): 
            A DataFrame containing prior knowledge of L-R-SPG interactions
                used to construct the prior knowledge graph. 
        gene_list (list): A list with length G. It is the gene name of gene expression.
        save_dir (str): save directory.
        curve_thred (float, optional): thredhold to filter out curves with low change magnitude.
            Default to 0.2
        p_thred (float, optional): thredhold for p_value for statistical significance
            Default to 0.01
        save_fig (bool, optional): Whether output L-R-SPG time series curves (only show inferred interactions). 
            Defaults to False.
        time_unit (str, optional): Time unit of the time axis.
            Defaults to None
        time_scale (float,optional): Scale of the time given in `time`. For example, time_scale=2: one unit in `time` is 2{time_unit}
            Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame, each row is a L-R-SPG pair, and its CCC causality is given in the DataFrame.
    """
    ## calculate CCC causality ##
    dt= (time[1]-time[0]) * time_scale
    # source_id_lst=[0,1,0,1]
    # target_id_lst=[1,0,0,1]
    # subfolder=['0to1','1to0','0to0','1to1']
    # lag_list = np.linspace(-2200,2200,441).astype(int)
    # data = {}
    # subfolder = str(source_id)+'to'+str(target_id)
    # # for idx in range(len(source_id)):

    # save_dir = dataset+'/'+subfolder+'/'
    os.makedirs(save_dir,exist_ok=True)
    data = {}
    for itm in df_LRSPG.iterrows():
        if itm[0]%10==0:
            print(itm[0])
        ligand_name = itm[1]['Ligand']
        SPG_name = itm[1]['SPG']
        receptor_name=itm[1]['Receptor']
        pathway = itm[1]['Pathway']
        signal_type=itm[1]['Signal Type']
        key=ligand_name+'-'+receptor_name+'-'+SPG_name
        
        l_idx=gene_list.index(ligand_name)
        r_idx=gene_list.index(receptor_name)
        spg_idx=gene_list.index(SPG_name)

        
        ligand = z_traj[source_id,:,l_idx]
        spg = z_traj[target_id,:,spg_idx]
        receptor = z_traj[target_id,:,r_idx]

        result = evaluate_causality(
            target = spg,
            ligand = ligand,
            # receptor=receptor,
            receptor=receptor,
            lag_list = lag_list,
            max_lag=max_lag,
            dt = dt,
            metrics='rmse',
            max_ligand=1.,
            max_receptor=1.,
            max_target = 1.,
            curve_thred=curve_thred,
            time_unit=time_unit,
            )
        result['pathway']=pathway
        result['signal type']=signal_type
        result['L']=ligand_name
        result['R']=receptor_name
        result['SPG']=SPG_name
        data[key] = result
        if result['p ftest']<p_thred or result['p chi2']<p_thred:
            if save_fig:
                textstr = '\n'.join((
                    f"Lag: {result['lag']:.3f}",
                    f"F test: {result['stat ftest']:.2f}, p={result['p ftest']:.2e}",
                    f"Chi2 test: {result['stat chi2']:.2f}, p={result['p chi2']:.2e}"
                    ))
                plt.plot(time*time_scale,ligand)
                plt.plot(time*time_scale,receptor)
                plt.plot(time*time_scale,spg)
                plt.legend(['L','R','SPG'])
                if time_unit is None:
                    xlabel='Time'
                else:
                    xlabel='Time ('+time_unit+')'
                plt.xlabel(xlabel)
                plt.ylabel('Exp')
                plt.title(key)
                plt.rcParams.update({'font.size': 16}) 
                plt.text(0.55, 0.2, textstr, fontsize=10,transform=plt.gca().transAxes,
                        verticalalignment='top')
                plt.ylim(0,1)
                plt.savefig(save_dir+key+'.pdf')
                plt.show()
                plt.close()
    
    ## FDR using Benjamini-Hochberg test
    p_ftest=[]
    p_chi2=[]
    key_list=list(data.keys())
    for key in key_list:
        p_ftest.append(data[key]['p ftest'])
        p_chi2.append(data[key]['p chi2'])
    p_ftest=np.array(p_ftest)
    p_chi2=np.array(p_chi2)
    q_ftest=false_discovery_control(p_ftest)
    q_chi2=false_discovery_control(p_chi2)
    for data_idx,key in enumerate(key_list):
        data[key]['q ftest']=q_ftest[data_idx]
        data[key]['q chi2']=q_chi2[data_idx]

    np.save(save_dir+'results.npy',data)
    for key in data:
        data[key].pop('rmse verbose', None)  
    df=pd.DataFrame(data).T
    df.to_csv(save_dir+'results.csv')
    return df
