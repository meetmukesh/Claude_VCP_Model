"""
Elite Swing Trading Scanner v5.0 - Database Integrated
Professional trading scanner with database persistence
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import sys

from database import DatabaseManager, ScanResult

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================
class ScannerConfig:
    """Elite configuration management"""

    # Database Settings
    DATABASE_URL = 'sqlite:///trading_scanner.db'  # Change to PostgreSQL/MySQL for production
    
    # Index Settings for Relative Strength
    BENCHMARK_INDEX = "^NSEI"  # Nifty 50
    BROAD_INDEX = "^CRSLDX"    # Nifty 500
    
    # Consolidation Parameters
    CONSOLIDATION_DAYS = 8
    MAX_CONSOLIDATION_RANGE = 0.06
    MIN_CONSOLIDATION_DAYS = 8
    MAX_CONSOLIDATION_DAYS = 15
    
    # Volume Parameters
    VOL_CONTRACTION_FACTOR = 0.70
    VOL_SURGE_THRESHOLD = 1.5
    MIN_AVG_DAILY_VOLUME = 500000
    
    # Trend Parameters
    MA_FAST = 21
    MA_SLOW = 50
    MA_TREND = 200
    
    # Momentum Parameters
    RSI_MIN = 40
    RSI_MAX = 70
    RSI_PERIOD = 14
    SWING_HIGH_LOOKBACK = 60
    SWING_HIGH_THRESHOLD = 0.08
    
    # Risk Parameters
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 2.0
    MIN_PRICE = 50
    MAX_PRICE = 5000
    
    # Relative Strength Parameters
    RS_3M_MIN = 1.05
    RS_6M_MIN = 1.10
    RS_WEIGHT_3M = 0.6
    RS_WEIGHT_6M = 0.4
    
    # VCP Parameters
    VCP_ENABLE = True
    VCP_STAGES = 3
    
    # Resistance
    RESISTANCE_PERIOD = 20
    BREAKOUT_PROXIMITY = 2.0
    
    # Data Parameters
    LOOKBACK_DAYS = 250
    MIN_DAILY_BARS = 150
    
    # Scoring
    ENABLE_REGIME_SCORING = True
    MIN_SCORE = 70


# =====================================================
# MARKET REGIME DETECTION
# =====================================================
class MarketRegime:
    """Detect market regime for adaptive scoring"""
    
    @staticmethod
    def detect_regime(index_data: pd.DataFrame) -> str:
        """Detect if market is in Bull, Bear, or Neutral regime"""
        if len(index_data) < 50:
            return 'NEUTRAL'
        
        current_price = index_data['Close'].iloc[-1]
        ma_50 = index_data['Close'].rolling(50).mean().iloc[-1]
        ma_200 = index_data['Close'].rolling(200).mean().iloc[-1]
        
        # Bull: Price > 50MA > 200MA and 50MA rising
        if current_price > ma_50 > ma_200:
            ma_50_slope = ma_50 - index_data['Close'].rolling(50).mean().iloc[-10]
            if ma_50_slope > 0:
                return 'BULL'
        
        # Bear: Price < 50MA < 200MA
        elif current_price < ma_50 < ma_200:
            return 'BEAR'
        
        return 'NEUTRAL'


# =====================================================
# INDEX DATA LOADER
# =====================================================
class IndexDataLoader:
    """Load and cache index data for RS calculations"""
    
    _cache = {}
    
    @staticmethod
    def load_index(symbol: str, start_date: datetime, db_manager: Optional[DatabaseManager] = None) -> pd.DataFrame:
        """Load index data with caching and database fallback"""
        
        # Check memory cache first
        if symbol in IndexDataLoader._cache:
            return IndexDataLoader._cache[symbol]
        
        # Try database if available
        if db_manager:
            df = db_manager.get_price_data(symbol, start_date)
            if not df.empty:
                IndexDataLoader._cache[symbol] = df
                return df
        
        # Download from Yahoo Finance
        try:
            df = yf.download(symbol, start=start_date, progress=False, timeout=10)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                
                # Save to database if available
                if db_manager:
                    db_manager.save_price_data(symbol, df)
                
                IndexDataLoader._cache[symbol] = df
                return df
        except Exception as e:
            logger.error(f"Failed to load index {symbol}: {e}")
        
        return pd.DataFrame()


# =====================================================
# TECHNICAL INDICATORS
# =====================================================
class TechnicalIndicators:
    """Elite technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_relative_strength(stock_df: pd.DataFrame, index_df: pd.DataFrame, 
                                   periods: Dict[str, int]) -> Dict[str, Optional[float]]:
        """Calculate Relative Strength vs Index"""
        rs_dict = {}
        
        for period_name, days in periods.items():
            if len(stock_df) < days or len(index_df) < days:
                rs_dict[period_name] = None
                continue
            
            stock_change = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-days]) - 1
            index_change = (index_df['Close'].iloc[-1] / index_df['Close'].iloc[-days]) - 1
            
            if index_change == 0:
                rs_dict[period_name] = None
            else:
                rs_dict[period_name] = (1 + stock_change) / (1 + index_change)
        
        return rs_dict
    
    @staticmethod
    def calculate_rs_slope(stock_df: pd.DataFrame, index_df: pd.DataFrame, 
                          lookback: int = 20) -> Tuple[Optional[float], Optional[float], bool]:
        """Calculate RS Slope - prevents RS decay traps"""
        if len(stock_df) < 63 + lookback or len(index_df) < 63 + lookback:
            return None, None, False
        
        # Current RS (3M)
        stock_change_now = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-63]) - 1
        index_change_now = (index_df['Close'].iloc[-1] / index_df['Close'].iloc[-63]) - 1
        rs_current = (1 + stock_change_now) / (1 + index_change_now) if index_change_now != 0 else None
        
        # RS 20 days ago
        stock_change_past = (stock_df['Close'].iloc[-lookback] / stock_df['Close'].iloc[-63-lookback]) - 1
        index_change_past = (index_df['Close'].iloc[-lookback] / index_df['Close'].iloc[-63-lookback]) - 1
        rs_past = (1 + stock_change_past) / (1 + index_change_past) if index_change_past != 0 else None
        
        is_improving = False
        if rs_current and rs_past:
            is_improving = rs_current > rs_past
        
        return rs_current, rs_past, is_improving
    
    @staticmethod
    def check_vcp(df: pd.DataFrame, lookback: int = 15) -> Tuple[bool, float]:
        """Check for Volatility Contraction Pattern"""
        if len(df) < lookback:
            return False, 0
        
        ranges = (df['High'] - df['Low']).rolling(5).mean()
        
        if len(ranges) < lookback:
            return False, 0
        
        stage_1 = ranges.iloc[-15:-10].mean()
        stage_2 = ranges.iloc[-10:-5].mean()
        stage_3 = ranges.iloc[-5:].mean()
        
        is_contracting = (stage_1 > stage_2 > stage_3)
        
        if is_contracting:
            contraction_ratio = stage_3 / stage_1 if stage_1 > 0 else 1
            quality = (1 - contraction_ratio) * 100
            return True, quality
        
        return False, 0
    
    @staticmethod
    def is_consolidating(df: pd.DataFrame, days_back: int = 15, max_range: float = 0.06) -> bool:
        """Check for tight consolidation"""
        if len(df) < days_back:
            return False
        
        recent = df['Close'].iloc[-days_back:]
        price_min = recent.min()
        
        if price_min == 0:
            return False
        
        range_pct = (recent.max() - price_min) / price_min
        return range_pct <= max_range


# =====================================================
# ELITE SCANNER ENGINE
# =====================================================
class EliteSwingScanner:
    """Elite swing trade scanner with database integration"""
    
    def __init__(self, config: ScannerConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.stats = self._initialize_stats()
        self.results = []
        self.market_regime = 'NEUTRAL'
        
        # Load index data
        start_date = datetime.today() - timedelta(days=config.LOOKBACK_DAYS + 50)
        logger.info("Loading benchmark indices...")
        
        self.nifty_data = IndexDataLoader.load_index(config.BENCHMARK_INDEX, start_date, db_manager)
        self.nifty500_data = IndexDataLoader.load_index(config.BROAD_INDEX, start_date, db_manager)
        
        if not self.nifty_data.empty:
            self.market_regime = MarketRegime.detect_regime(self.nifty_data)
            logger.info(f"Market Regime: {self.market_regime}")
        else:
            logger.warning("Could not load index data - RS calculations disabled")
    
    def _initialize_stats(self) -> Dict[str, int]:
        return {
            "total_scanned": 0,
            "data_load_failed": 0,
            "sufficient_data": 0,
            "price_filter": 0,
            "volume_filter": 0,
            "trend_filter": 0,
            "consolidation_filter": 0,
            "momentum_filter": 0,
            "rs_filter": 0,
            "rs_slope_filter": 0,
            "vcp_filter": 0,
            "near_breakout": 0,
            "high_probability": 0
        }
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare and validate dataframe"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return None
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df['MA21'] = df['Close'].rolling(self.config.MA_FAST).mean()
        df['MA50'] = df['Close'].rolling(self.config.MA_SLOW).mean()
        df['MA200'] = df['Close'].rolling(self.config.MA_TREND).mean()
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], self.config.RSI_PERIOD)
        df['ATR'] = TechnicalIndicators.calculate_atr(df, self.config.ATR_PERIOD)
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        return df
    
    def _apply_filters(self, ticker: str, df: pd.DataFrame) -> Optional[Dict]:
        """Apply all elite filters and return results"""
        
        current = df.iloc[-1]
        filter_results = {
            'symbol': ticker.replace('.NS', ''),
            'filters_passed': 0,
            'filter_details': {}
        }
        
        # Validation
        if pd.isna(current['ATR']) or current['ATR'] <= 0:
            return None
        
        # Filter 1: Price & Liquidity
        if current['Close'] < self.config.MIN_PRICE or current['Close'] > self.config.MAX_PRICE:
            filter_results['filter_details']['price'] = 'Failed'
            return filter_results
        
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        if pd.isna(avg_volume) or avg_volume < self.config.MIN_AVG_DAILY_VOLUME:
            filter_results['filter_details']['volume'] = 'Failed'
            return filter_results
        
        filter_results['filters_passed'] = 1
        filter_results['filter_details']['price_volume'] = 'Passed'
        self.stats["price_filter"] += 1
        
        # Filter 2: Volume Contraction
        vol_recent = df['Volume'].iloc[-5:].mean()
        vol_older = df['Volume'].iloc[-25:-5].mean()
        
        volume_contraction = False
        if vol_older > 0:
            vol_ratio = vol_recent / vol_older
            volume_contraction = vol_ratio < self.config.VOL_CONTRACTION_FACTOR
        
        if not volume_contraction:
            filter_results['filter_details']['vol_contraction'] = 'Failed'
            return filter_results
        
        filter_results['filters_passed'] = 2
        filter_results['filter_details']['vol_contraction'] = 'Passed'
        self.stats["volume_filter"] += 1
        
        # Filter 3: Trend Alignment
        if pd.isna(current['MA21']) or pd.isna(current['MA50']) or pd.isna(current['MA200']):
            filter_results['filter_details']['trend'] = 'Insufficient Data'
            return filter_results
        
        trend_aligned = current['Close'] > current['MA21'] > current['MA50']
        
        if not trend_aligned:
            filter_results['filter_details']['trend'] = 'Failed'
            return filter_results
        
        filter_results['filters_passed'] = 3
        filter_results['filter_details']['trend'] = 'Passed'
        self.stats["trend_filter"] += 1
        
        # Filter 4: Consolidation
        is_consolidating = TechnicalIndicators.is_consolidating(
            df, self.config.CONSOLIDATION_DAYS, self.config.MAX_CONSOLIDATION_RANGE
        )
        
        if not is_consolidating:
            filter_results['filter_details']['consolidation'] = 'Failed'
            return filter_results
        
        consol_range = self._calculate_consolidation_range(df, self.config.CONSOLIDATION_DAYS)
        filter_results['filters_passed'] = 4
        filter_results['filter_details']['consolidation'] = 'Passed'
        filter_results['consolidation_range'] = consol_range
        self.stats["consolidation_filter"] += 1
        
        # Filter 5: Momentum (RSI)
        if pd.isna(current['RSI']):
            filter_results['filter_details']['momentum'] = 'Insufficient Data'
            return filter_results
        
        rsi_valid = self.config.RSI_MIN <= current['RSI'] <= self.config.RSI_MAX
        
        if not rsi_valid:
            filter_results['filter_details']['momentum'] = 'Failed'
            return filter_results
        
        filter_results['filters_passed'] = 5
        filter_results['filter_details']['momentum'] = 'Passed'
        filter_results['rsi'] = current['RSI']
        self.stats["momentum_filter"] += 1
        
        # Filter 6: Relative Strength
        if not self.nifty_data.empty:
            rs_dict = TechnicalIndicators.calculate_relative_strength(
                df, self.nifty_data, {'3M': 63, '6M': 126}
            )
            
            rs_3m = rs_dict.get('3M')
            rs_6m = rs_dict.get('6M')
            
            if rs_3m is None or rs_6m is None:
                filter_results['filter_details']['relative_strength'] = 'Insufficient Data'
                return filter_results
            
            rs_qualified = (rs_3m >= self.config.RS_3M_MIN and rs_6m >= self.config.RS_6M_MIN)
            
            if not rs_qualified:
                filter_results['filter_details']['relative_strength'] = 'Failed'
                return filter_results
            
            filter_results['filters_passed'] = 6
            filter_results['filter_details']['relative_strength'] = 'Passed'
            filter_results['rs_3m'] = rs_3m
            filter_results['rs_6m'] = rs_6m
            filter_results['rs_composite'] = (rs_3m * self.config.RS_WEIGHT_3M + 
                                             rs_6m * self.config.RS_WEIGHT_6M)
            self.stats["rs_filter"] += 1
            
            # Filter 7: RS Slope (improving RS)
            rs_current, rs_past, is_improving = TechnicalIndicators.calculate_rs_slope(df, self.nifty_data)
            
            if not is_improving:
                filter_results['filter_details']['rs_slope'] = 'Failed'
                return filter_results
            
            filter_results['filters_passed'] = 7
            filter_results['filter_details']['rs_slope'] = 'Passed'
            self.stats["rs_slope_filter"] += 1
        else:
            # Skip RS filters if no index data
            filter_results['filters_passed'] = 5
            filter_results['filter_details']['relative_strength'] = 'Skipped - No Index Data'
            filter_results['filter_details']['rs_slope'] = 'Skipped - No Index Data'
        
        # Filter 8: VCP (optional but scoring)
        if self.config.VCP_ENABLE:
            is_vcp, vcp_quality = TechnicalIndicators.check_vcp(df)
            filter_results['vcp_qualified'] = is_vcp
            filter_results['vcp_quality'] = vcp_quality
            
            if is_vcp:
                self.stats["vcp_filter"] += 1
        
        # Calculate resistance and proximity
        resistance = df['Close'].iloc[-self.config.RESISTANCE_PERIOD:].max()
        distance_to_resistance = ((resistance - current['Close']) / current['Close']) * 100
        
        near_breakout = distance_to_resistance <= self.config.BREAKOUT_PROXIMITY
        
        if near_breakout:
            self.stats["near_breakout"] += 1
        
        filter_results['resistance'] = resistance
        filter_results['distance_to_resistance'] = distance_to_resistance
        filter_results['near_breakout'] = near_breakout
        
        # Calculate stop loss
        stop_loss = current['Close'] - (current['ATR'] * self.config.ATR_MULTIPLIER)
        risk_pct = ((current['Close'] - stop_loss) / current['Close']) * 100
        
        filter_results['stop_loss'] = stop_loss
        filter_results['risk_pct'] = risk_pct
        
        # Calculate score
        score = self._calculate_score(filter_results, current, df)
        filter_results['total_score'] = score
        
        # Price data
        filter_results['current_price'] = current['Close']
        filter_results['volume'] = current['Volume']
        filter_results['avg_volume_20d'] = avg_volume
        filter_results['ma_21'] = current['MA21']
        filter_results['ma_50'] = current['MA50']
        filter_results['ma_200'] = current['MA200']
        filter_results['atr_14'] = current['ATR']
        
        # Calculate price change
        if len(df) > 1:
            prev_close = df['Close'].iloc[-2]
            filter_results['price_change_pct'] = ((current['Close'] - prev_close) / prev_close) * 100
        
        filter_results['market_regime'] = self.market_regime
        
        # Check if fully qualified
        min_filters = 7 if not self.nifty_data.empty else 5
        filter_results['is_qualified'] = (filter_results['filters_passed'] >= min_filters and 
                                          score >= self.config.MIN_SCORE)
        
        if filter_results['is_qualified']:
            self.stats["high_probability"] += 1
        
        return filter_results
    
    def _calculate_consolidation_range(self, df: pd.DataFrame, days: int) -> float:
        """Calculate consolidation range percentage"""
        recent = df['Close'].iloc[-days:]
        price_min = recent.min()
        if price_min == 0:
            return 0
        return ((recent.max() - price_min) / price_min) * 100
    
    def _calculate_score(self, filters: Dict, current: pd.Series, df: pd.DataFrame) -> float:
        """Calculate total score for the stock"""
        score = 0
        
        # Base filters (60 points)
        score += filters['filters_passed'] * 8.57  # ~60 points for 7 filters
        
        # Bonus: VCP Pattern (10 points)
        if filters.get('vcp_qualified'):
            score += 10
        
        # Bonus: Near Breakout (10 points)
        if filters.get('near_breakout'):
            score += 10
        
        # Bonus: Strong RS (10 points)
        if filters.get('rs_composite', 0) >= 1.15:
            score += 10
        
        # Bonus: Tight Consolidation (5 points)
        if filters.get('consolidation_range', 100) < 3:
            score += 5
        
        # Bonus: Low Risk (5 points)
        if filters.get('risk_pct', 100) < 5:
            score += 5
        
        return min(score, 100)
    
    def scan_stock(self, ticker: str) -> Optional[Dict]:
        """Scan a single stock"""
        self.stats["total_scanned"] += 1
        
        try:
            # Try to get from database first
            start_date = datetime.today() - timedelta(days=self.config.LOOKBACK_DAYS)
            df = self.db.get_price_data(ticker, start_date)
            
            # If not in DB or stale, download
            if df.empty or (datetime.now() - df.index[-1]).days > 1:
                logger.debug(f"Downloading fresh data for {ticker}")
                df = yf.download(ticker, start=start_date, progress=False, timeout=10)
                
                if not df.empty:
                    df = self._prepare_dataframe(df)
                    if df is not None:
                        # Save to database
                        self.db.save_price_data(ticker, df)
            
            if df.empty or df is None:
                self.stats["data_load_failed"] += 1
                return None
            
            if len(df) < self.config.MIN_DAILY_BARS:
                return None
            
            self.stats["sufficient_data"] += 1
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Apply filters
            result = self._apply_filters(ticker, df)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
            self.stats["data_load_failed"] += 1
            return None
    
    def scan_multiple(self, tickers: List[str]) -> pd.DataFrame:
        """Scan multiple stocks and return results"""
        logger.info(f"Starting scan of {len(tickers)} stocks...")
        
        results = []
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} stocks scanned")
            
            result = self.scan_stock(ticker)
            if result:
                results.append(result)
        
        if not results:
            logger.warning("No stocks passed any filters")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('total_score', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SCAN COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Scanned: {self.stats['total_scanned']}")
        logger.info(f"Data Load Failed: {self.stats['data_load_failed']}")
        logger.info(f"Sufficient Data: {self.stats['sufficient_data']}")
        logger.info(f"Passed Price Filter: {self.stats['price_filter']}")
        logger.info(f"Passed Volume Filter: {self.stats['volume_filter']}")
        logger.info(f"Passed Trend Filter: {self.stats['trend_filter']}")
        logger.info(f"Passed Consolidation: {self.stats['consolidation_filter']}")
        logger.info(f"Passed Momentum: {self.stats['momentum_filter']}")
        logger.info(f"Passed RS Filter: {self.stats['rs_filter']}")
        logger.info(f"Passed RS Slope: {self.stats['rs_slope_filter']}")
        logger.info(f"VCP Patterns Found: {self.stats['vcp_filter']}")
        logger.info(f"Near Breakout: {self.stats['near_breakout']}")
        logger.info(f"HIGH PROBABILITY SETUPS: {self.stats['high_probability']}")
        logger.info(f"{'='*60}\n")
        
        return df
    
    def get_stats(self) -> Dict[str, int]:
        """Get scan statistics"""
        return self.stats.copy()


def main():
    """Main execution function"""
    # Initialize database
    db = DatabaseManager()
    db.create_tables()
    
    # Load stock universe
    logger.info("Loading stock universe...")
    symbols = db.get_active_stocks()
    
    if not symbols:
        logger.info("No stocks in database. Please load stock universe first.")
        return
    
    # Add exchange suffix
    tickers = [f"{symbol}.NS" for symbol in symbols]
    
    # Initialize scanner
    config = ScannerConfig()
    scanner = EliteSwingScanner(config, db)
    
    # Run scan
    start_time = datetime.now()
    results_df = scanner.scan_multiple(tickers)
    end_time = datetime.now()
    
    # Save results to database
    if not results_df.empty:
        scan_date = datetime.now()
        db.save_scan_results(results_df, scan_date)
        
        # Save metadata
        stats = scanner.get_stats()
        db.save_scan_metadata(
            scan_type='FULL_SCAN',
            start_time=start_time,
            end_time=end_time,
            stats=stats,
            market_regime=scanner.market_regime,
            config_snapshot=vars(config),
            status='SUCCESS'
        )
        
        # Export qualified stocks
        qualified = results_df[results_df['is_qualified'] == True]
        if not qualified.empty:
            filepath = f"qualified_stocks_{scan_date.strftime('%Y%m%d')}.csv"
            qualified.to_csv(filepath, index=False)
            logger.info(f"Exported {len(qualified)} qualified stocks to {filepath}")
    
    logger.info("Scan completed successfully!")


if __name__ == "__main__":
    main()
