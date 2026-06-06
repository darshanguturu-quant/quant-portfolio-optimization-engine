# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize as sc_minimize
import plotly.graph_objects as go
import datetime as dt

# Function to fetch data using yfinance
def getData(stocks, start, end):
    print(f"\nFetching data for: {', '.join(stocks)}")
    try:
        stockData = yf.download(stocks, start=start, end=end)['Close']
        if stockData.empty:
            raise ValueError("No data fetched. Check ticker symbols and date range.")
        # Forward fill and drop remaining NaNs
        stockData = stockData.fillna(method='ffill').dropna()
        print("Data fetched successfully.")
        returns = stockData.pct_change().dropna()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix
    except Exception as e:
        print(f"Error fetching data: {e}")
        exit()

# Function to calculate portfolio performance
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * 252  # Annualize return
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)  # Annualize std dev
    return returns, std

# Function to minimize negative Sharpe Ratio
def negativeSR(weights, meanReturns, covMatrix, riskFreeRate):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -( (pReturns - riskFreeRate) / pStd )

# Function to maximize Sharpe Ratio
def maxSR(meanReturns, covMatrix, riskFreeRate, constraintSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    print("Optimizing for Maximum Sharpe Ratio...")
    result = sc_minimize(
        negativeSR,
        numAssets * [1. / numAssets],
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        print(f"Optimization failed: {result.message}")
        raise BaseException(result.message)
    print("Optimization for Maximum Sharpe Ratio successful.")
    return result

# Function to calculate portfolio variance
def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

# Function to minimize portfolio variance
def minimizeVariance(meanReturns, covMatrix, constraintSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    print("Optimizing for Minimum Volatility...")
    result = sc_minimize(
        portfolioVariance,
        numAssets * [1. / numAssets],
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        print(f"Optimization failed: {result.message}")
        raise BaseException(result.message)
    print("Optimization for Minimum Volatility successful.")
    return result

# Function to calculate portfolio return (used for Efficient Frontier)
def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

# Function to optimize for Efficient Frontier
def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, optimize the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = (
        {'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = tuple(constraintSet for asset in range(numAssets))
    try:
        effOpt = sc_minimize(
            portfolioVariance,
            numAssets*[1./numAssets],
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if effOpt.success:
            return effOpt
        else:
            print(f"Efficient Frontier optimization failed for target return {returnTarget}: {effOpt.message}")
            return None
    except Exception as e:
        print(f"Exception during Efficient Frontier optimization for target return {returnTarget}: {e}")
        return None

# Function to calculate Efficient Frontier results
def calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet=(0,1)):
    """Calculate Max SR, Min Volatility, and Efficient Frontier"""
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio.x, meanReturns, covMatrix)
    
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio.x, index=meanReturns.index, columns=['Allocation (%)'])
    maxSR_allocation['Allocation (%)'] = [round(i*100,2) for i in maxSR_allocation['Allocation (%)']]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix, constraintSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio.x, meanReturns, covMatrix)
    
    minVol_allocation = pd.DataFrame(minVol_Portfolio.x, index=meanReturns.index, columns=['Allocation (%)'])
    minVol_allocation['Allocation (%)'] = [round(i*100,2) for i in minVol_allocation['Allocation (%)']]

    # Efficient Frontier
    print("Calculating Efficient Frontier...")
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 100)  # Increased points for smoother frontier
    for target in targetReturns:
        eff = efficientOpt(meanReturns, covMatrix, target, constraintSet)
        if eff is not None:
            eff_std = portfolioPerformance(eff.x, meanReturns, covMatrix)[1]
            efficientList.append(eff_std)
        else:
            efficientList.append(np.nan)
    print("Efficient Frontier calculation completed.")
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

# Function to plot Efficient Frontier using Plotly
def EF_graph_plotly(meanReturns, covMatrix, riskFreeRate, maxSR_returns, maxSR_std, constraintSet=(0,1), benchmarkPoints=None):
    """Plot Efficient Frontier with Max SR, Min Volatility Portfolios, CML, and Benchmarks using Plotly"""
    
    # Calculate Efficient Frontier and get portfolio metrics
    maxSR_returns, maxSR_std, _, minVol_returns, minVol_std, _, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    
    # Convert to percentages
    efficient_std_pct = [std * 100 for std in efficientList]
    target_returns_pct = [ret * 100 for ret in targetReturns]
    maxSR_std_pct = maxSR_std * 100
    maxSR_returns_pct = maxSR_returns * 100
    minVol_std_pct = minVol_std * 100
    minVol_returns_pct = minVol_returns * 100
    
    # Calculate Capital Market Line (CML)
    cml_x = [0, maxSR_std_pct]
    cml_y = [riskFreeRate * 100, maxSR_returns_pct]
    
    # Create traces
    ef_trace = go.Scatter(
        x=efficient_std_pct,
        y=target_returns_pct,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='black', width=4, dash='dashdot')
    )
    
    maxSR_trace = go.Scatter(
        x=[maxSR_std_pct],
        y=[maxSR_returns_pct],
        mode='markers',
        name='Max Sharpe Ratio',
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )
    
    minVol_trace = go.Scatter(
        x=[minVol_std_pct],
        y=[minVol_returns_pct],
        mode='markers',
        name='Min Volatility',
        marker=dict(color='blue', size=14, line=dict(width=3, color='black'))
    )
    
    cml_trace = go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines',
        name='Capital Market Line (CML)',
        line=dict(color='green', width=2, dash='dash')
    )
    
    # Benchmark Points
    benchmark_traces = []
    if benchmarkPoints:
        for benchmark in benchmarkPoints:
            benchmark_traces.append(go.Scatter(
                x=[benchmark['std'] * 100],
                y=[benchmark['mean'] * 100],
                mode='markers',
                name=f"Benchmark: {benchmark['ticker']}",
                marker=dict(symbol='diamond-open', size=12, line=dict(width=2, color='black'))
            ))
    
    # Plot
    fig = go.Figure(data=[ef_trace, maxSR_trace, minVol_trace, cml_trace] + benchmark_traces)
    fig.update_layout(
        title='Portfolio Optimization with Efficient Frontier and CML',
        xaxis_title='Annualized Volatility (%)',
        yaxis_title='Annualized Return (%)',
        showlegend=True,
        legend=dict(
            x=0.75, y=0.25,
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2
        ),
        width=900,
        height=700
    )
    fig.show()

# Function to fetch benchmark data and calculate performance
def getBenchmarkPerformance(indexTicker, start, end, riskFreeRate):
    print(f"Fetching benchmark data for: {indexTicker}")
    try:
        indexData = yf.download(indexTicker, start=start, end=end)['Close']
        if indexData.empty:
            raise ValueError("No data fetched for benchmark. Check ticker symbol and date range.")
        indexReturns = indexData.pct_change().dropna()
        meanReturn = indexReturns.mean() * 252  # Annualized return
        stdDev = indexReturns.std() * np.sqrt(252)  # Annualized std dev
        sharpe = (meanReturn - riskFreeRate) / stdDev
        print(f"Benchmark {indexTicker} fetched successfully.")
        return meanReturn, stdDev, sharpe
    except Exception as e:
        print(f"Error fetching benchmark data for {indexTicker}: {e}")
        return None, None, None

# Function to display portfolio allocations
def displayAllocations(title, allocation_df):
    print(f"\n{title}:")
    for asset, weight in zip(allocation_df.index, allocation_df['Allocation (%)']):
        print(f"{asset}: {weight}%")

# Function to run portfolio analysis for each phase and scenario
def runPortfolioAnalysis(phaseName, stockList, riskFreeRate, benchmarkTickers, benchmarkPortfolios, startDate, endDate, allowShortSelling):
    print(f"\n=== {phaseName} ===")
    print(f"Assets: {', '.join(stockList)}")
    print(f"Short Selling Allowed: {'Yes' if allowShortSelling else 'No'}")
    
    # Set constraints based on short selling allowance
    if allowShortSelling:
        constraintSet = (-1, 1)  # Allowing up to 100% short selling
    else:
        constraintSet = (0, 1)   # No short selling
    
    # Fetch data
    meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
    
    # Optimize for Max Sharpe Ratio
    maxSR_result = maxSR(meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_result.x, meanReturns, covMatrix)
    maxSR_sharpe = (maxSR_returns - riskFreeRate) / maxSR_std
    
    maxSR_allocation = pd.DataFrame(maxSR_result.x, index=meanReturns.index, columns=['Allocation (%)'])
    maxSR_allocation['Allocation (%)'] = [round(i*100,2) for i in maxSR_allocation['Allocation (%)']]
    
    # Optimize for Min Variance
    minVol_result = minimizeVariance(meanReturns, covMatrix, constraintSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_result.x, meanReturns, covMatrix)
    minVol_sharpe = (minVol_returns - riskFreeRate) / minVol_std
    
    minVol_allocation = pd.DataFrame(minVol_result.x, index=meanReturns.index, columns=['Allocation (%)'])
    minVol_allocation['Allocation (%)'] = [round(i*100,2) for i in minVol_allocation['Allocation (%)']]
    
    # Display Results
    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    print(f"Expected Annual Return: {maxSR_returns*100:.2f}%")
    print(f"Annual Volatility (Std Dev): {maxSR_std*100:.2f}%")
    print(f"Sharpe Ratio: {maxSR_sharpe:.4f}")
    displayAllocations("Asset Allocations", maxSR_allocation)
    
    print("\n--- Minimum Volatility Portfolio ---")
    print(f"Expected Annual Return: {minVol_returns*100:.2f}%")
    print(f"Annual Volatility (Std Dev): {minVol_std*100:.2f}%")
    print(f"Sharpe Ratio: {minVol_sharpe:.4f}")
    displayAllocations("Asset Allocations", minVol_allocation)
    
    # Prepare Benchmark Points for Plotting
    benchmark_points = []
    # Fetch benchmark tickers performances
    for benchmark in benchmarkTickers:
        mean_bench, std_bench, sharpe_bench = getBenchmarkPerformance(benchmark, startDate, endDate, riskFreeRate)
        if mean_bench is not None:
            benchmark_points.append({'ticker': benchmark, 'mean': mean_bench, 'std': std_bench})
    # Add benchmark portfolios
    for benchmark in benchmarkPortfolios:
        benchmark_points.append({'ticker': benchmark['name'], 'mean': benchmark['mean'], 'std': benchmark['std']})
    
    # Plot Efficient Frontier with Benchmarks using Plotly
    print("Plotting Efficient Frontier with Plotly...")
    EF_graph_plotly(meanReturns, covMatrix, riskFreeRate, maxSR_returns, maxSR_std, constraintSet, benchmarkPoints=benchmark_points)
    print("Efficient Frontier plotted successfully with Plotly.")
    
    # Compare Portfolio with Benchmark
    print("\n--- Portfolio vs Benchmark ---")
    for benchmark in benchmarkTickers:
        mean_bench, std_bench, sharpe_bench = getBenchmarkPerformance(benchmark, startDate, endDate, riskFreeRate)
        if mean_bench is not None:
            print(f"\nMax Sharpe Ratio Portfolio vs {benchmark}:")
            print(f"  Return: {maxSR_returns*100:.2f}% vs {mean_bench*100:.2f}%")
            print(f"  Volatility: {maxSR_std*100:.2f}% vs {std_bench*100:.2f}%")
            print(f"  Sharpe Ratio: {maxSR_sharpe:.4f} vs {sharpe_bench:.4f}")
            
            print(f"\nMin Volatility Portfolio vs {benchmark}:")
            print(f"  Return: {minVol_returns*100:.2f}% vs {mean_bench*100:.2f}%")
            print(f"  Volatility: {minVol_std*100:.2f}% vs {std_bench*100:.2f}%")
            print(f"  Sharpe Ratio: {minVol_sharpe:.4f} vs {sharpe_bench:.4f}")
    
    for benchmark in benchmarkPortfolios:
        print(f"\nMax Sharpe Ratio Portfolio vs {benchmark['name']}:")
        print(f"  Return: {maxSR_returns*100:.2f}% vs {benchmark['mean']*100:.2f}%")
        print(f"  Volatility: {maxSR_std*100:.2f}% vs {benchmark['std']*100:.2f}%")
        print(f"  Sharpe Ratio: {maxSR_sharpe:.4f} vs {(benchmark['mean'] - riskFreeRate) / benchmark['std']:.4f}")
        
        print(f"\nMin Volatility Portfolio vs {benchmark['name']}:")
        print(f"  Return: {minVol_returns*100:.2f}% vs {benchmark['mean']*100:.2f}%")
        print(f"  Volatility: {minVol_std*100:.2f}% vs {benchmark['std']*100:.2f}%")
        print(f"  Sharpe Ratio: {minVol_sharpe:.4f} vs {(benchmark['mean'] - riskFreeRate) / benchmark['std']:.4f}")
    
    # Return portfolio performances for benchmarking in subsequent phases
    maxSR_performance = {'name': f"{phaseName}_MaxSR", 'mean': maxSR_returns, 'std': maxSR_std}
    minVol_performance = {'name': f"{phaseName}_MinVol", 'mean': minVol_returns, 'std': minVol_std}
    
    return maxSR_performance, minVol_performance

# Example usage:
if __name__ == "__main__":
    
    # Set the risk-free rate to 6%
    riskFreeRate = 0.06  # 6%
    
    # Define the date range for data fetching
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365)  # Past year
    
    # Define Benchmark Tickers
    benchmark_phase1 = ['^NSEI']  # NIFTY50
    # For Phase2 and Phase3, benchmarks will include Phase1's portfolios
    
    # Define Phase 1: Only 8 Domestic Stocks
    stockList_phase1 = [
        'SUNDRMFAST.NS', 'JKCEMENT.NS', 'RBLBANK.NS', 'HINDPETRO.NS',
        'SANOFI.NS', 'RAYMOND.NS', 'MEDPLUS.NS', 'RELIANCE.NS'
    ]
    
    # Define Phase 2: 8 Domestic + 1 International Security
    international_stock_phase2 = ['AAPL']  # Apple Inc.
    stockList_phase2 = stockList_phase1 + international_stock_phase2  # 9 assets
    
    # Define Phase 3: 8 Domestic + 1 International + 1 Cryptocurrency
    cryptocurrency_phase3 = ['BTC-USD']  # Bitcoin
    stockList_phase3 = stockList_phase2 + cryptocurrency_phase3  # 10 assets
    
    # Define short selling scenarios
    scenarios = [
        {'allowShortSelling': False, 'description': "No Short Selling"},
        {'allowShortSelling': True, 'description': "Short Selling Allowed"}
    ]
    
    # Initialize a dictionary to store Phase1's portfolios per scenario
    phase1_portfolios = {
        "No Short Selling": {},
        "Short Selling Allowed": {}
    }
    
    # Iterate over each scenario
    for scenario in scenarios:
        allowShortSelling = scenario['allowShortSelling']
        description = scenario['description']
        
        print(f"\n=== Scenario: {description} ===")
        
        # Phase 1: Only 8 Domestic Stocks compared with NIFTY50
        phase1_name = "Phase 1: Only 8 Domestic Stocks"
        phase1_benchmarks = benchmark_phase1  # NIFTY50 only
        phase1_portfolio = runPortfolioAnalysis(
            phaseName=phase1_name,
            stockList=stockList_phase1,
            riskFreeRate=riskFreeRate,
            benchmarkTickers=phase1_benchmarks,
            benchmarkPortfolios=[],  # No additional benchmarks
            startDate=startDate,
            endDate=endDate,
            allowShortSelling=allowShortSelling
        )
        
        # Store Phase1's portfolios for benchmarking in Phase2 and Phase3
        phase1_portfolios[description]['MaxSR'] = phase1_portfolio[0]
        phase1_portfolios[description]['MinVol'] = phase1_portfolio[1]
        
        # Phase 2: 8 Domestic + 1 International Security compared with Phase1 and NIFTY50
        phase2_name = "Phase 2: 8 Domestic + 1 International Security"
        # Benchmarks: NIFTY50 and Phase1's MaxSR & MinVol
        phase2_benchmarks = benchmark_phase1.copy()  # NIFTY50
        phase2_additional_benchmarks = [
            phase1_portfolios[description]['MaxSR'],
            phase1_portfolios[description]['MinVol']
        ]
        phase2_portfolio = runPortfolioAnalysis(
            phaseName=phase2_name,
            stockList=stockList_phase2,
            riskFreeRate=riskFreeRate,
            benchmarkTickers=phase2_benchmarks,
            benchmarkPortfolios=phase2_additional_benchmarks,  # Phase1's Portfolios
            startDate=startDate,
            endDate=endDate,
            allowShortSelling=allowShortSelling
        )
        
        # Phase 3: 8 Domestic + 1 International + 1 Cryptocurrency compared with Phase1 and NIFTY50
        phase3_name = "Phase 3: 8 Domestic + 1 International + 1 Cryptocurrency"
        # Benchmarks: NIFTY50 and Phase1's MaxSR & MinVol
        phase3_benchmarks = benchmark_phase1.copy()  # NIFTY50
        phase3_additional_benchmarks = [
            phase1_portfolios[description]['MaxSR'],
            phase1_portfolios[description]['MinVol']
        ]
        phase3_portfolio = runPortfolioAnalysis(
            phaseName=phase3_name,
            stockList=stockList_phase3,
            riskFreeRate=riskFreeRate,
            benchmarkTickers=phase3_benchmarks,
            benchmarkPortfolios=phase3_additional_benchmarks,  # Phase1's Portfolios
            startDate=startDate,
            endDate=endDate,
            allowShortSelling=allowShortSelling
        )
    
    print("\n=== Portfolio Analysis Completed ===")
