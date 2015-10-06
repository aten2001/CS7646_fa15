"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from util import get_data, plot_data


def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """
    port_val = start_val
    l_port_vals_by_day = []
    
    for idx in range(len(prices)):
        if idx == 0:
            port_val_this_day = start_val
            l_port_vals_by_day.append(port_val_this_day)
        else:
            last_port_val = l_port_vals_by_day[idx-1]
            today_stock_vals = []
            for idx_stock in range(len(allocs)):
                alloc= allocs[idx_stock]
                val_this_stock = (prices.iloc[idx][idx_stock] / prices.iloc[idx-1][idx_stock] ) * alloc * last_port_val
                today_stock_vals.append(val_this_stock)
            port_val_this_day = np.sum(today_stock_vals)
            l_port_vals_by_day.append(port_val_this_day)
    df_port_vals = pd.DataFrame([])
    df_port_vals[0] = l_port_vals_by_day
    df_port_vals.index = prices.index
    return df_port_vals


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # TODO: Your code here
    cum_ret = list(port_val[0])[-1] - list(port_val[0])[0]
    
    l_daily_ret = []
    for idx in range(len(port_val)):
        if idx == 0:
            l_daily_ret.append(0)
        else:
            return_this_day = (port_val.iloc[idx][0] - port_val.iloc[idx-1][0]) / port_val.iloc[idx-1][0]
            l_daily_ret.append(return_this_day)
    cum_ret = np.sum(l_daily_ret)
    avg_daily_ret = np.mean(l_daily_ret)
    std_daily_ret = np.std(l_daily_ret)
    sharpe_ratio = (avg_daily_ret - daily_rf) / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    #TODO: Your code here

    ldt_timestamps = list(df.index)  # timestamps from dataframe
    ls_symbols = list(df.columns.values) # symbols for the stocks
    
    df_normalized = df.copy()
    df_normalized[ls_symbols] = df_normalized[ls_symbols].apply(lambda x: (x - x[0]) / x[0] + 1)

    na_normalized_price = df_normalized.values

    # print na_normalized_price[0:10][:]
    # for i in range(len(na_normalized_price[0])):
    #     na_normalized_price[:][i] = (na_normalized_price[:][i] - na_normalized_price[0][i]) / na_normalized_price[0][i]

    # print na_normalized_price[0:10][:]

    plt.clf()
    plt.plot(ldt_timestamps, na_normalized_price)
    plt.legend(ls_symbols)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('comparison.png', format='png')

def assess_portfolio(start_date, end_date, symbols, allocs, start_val=1):
    """Simulate and assess the performance of a stock portfolio."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, start_val)
    #plot_data(port_val, title="Daily Portfolio Value")

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with SPY using a normalized plot
    df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and SPY")


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'

    symbol_allocations = OrderedDict([('GOOG', 0.2), ('AAPL', 0.3), ('GLD', 0.4), ('XOM', 0.1)])  # symbols and corresponding allocations
    #symbol_allocations = OrderedDict([('AXP', 0.0), ('HPQ', 0.0), ('IBM', 0.0), ('HNZ', 1.0)])  # allocations from wiki example

    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  # starting value of portfolio

    # Assess the portfolio
    assess_portfolio(start_date, end_date, symbols, allocs, start_val)


if __name__ == "__main__":
    test_run()
