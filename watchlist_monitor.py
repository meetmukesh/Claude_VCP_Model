"""
Professional Watchlist Monitor System
Tracks "Almost Qualified" Stocks Through Filter Progression
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import yfinance as yf

from database import DatabaseManager
from scanner import EliteSwingScanner, ScannerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatchlistConfig:
    """Watchlist monitoring configuration"""
    
    # Minimum filters to track
    MIN_FILTERS_PASSED = 4
    
    # Alert thresholds
    ALERT_ON_FILTER_IMPROVEMENT = True
    ALERT_ON_QUALIFICATION = True
    ALERT_ON_DETERIORATION = True
    ALERT_ON_REMOVAL = True
    
    # Deterioration threshold
    DETERIORATION_THRESHOLD = 2  # Remove if drops 2+ filters
    
    # Auto-removal
    AUTO_REMOVE_QUALIFIED_AFTER_DAYS = 30
    AUTO_REMOVE_DETERIORATED_AFTER_DAYS = 7


class WatchlistMonitor:
    """Monitor and track stocks progressing through filters"""
    
    def __init__(self, db_manager: DatabaseManager, config: WatchlistConfig):
        self.db = db_manager
        self.config = config
        self.scanner_config = ScannerConfig()
        self.scanner = EliteSwingScanner(self.scanner_config, db_manager)
    
    def update_watchlist_from_scan(self, scan_results_df: pd.DataFrame):
        """Update watchlist based on latest scan results"""
        logger.info("Updating watchlist from scan results...")
        
        additions = 0
        updates = 0
        alerts_created = 0
        
        for _, row in scan_results_df.iterrows():
            symbol = row['symbol']
            filters_passed = row['filters_passed']
            score = row['total_score']
            price = row['current_price']
            stop_loss = row['stop_loss']
            is_qualified = row.get('is_qualified', False)
            
            # Check if stock should be tracked
            if filters_passed >= self.config.MIN_FILTERS_PASSED or is_qualified:
                
                # Get existing watchlist entry
                existing = self._get_watchlist_entry(symbol)
                
                if existing is None:
                    # Add new entry
                    self.db.update_watchlist(
                        symbol=symbol,
                        filters_passed=filters_passed,
                        score=score,
                        price=price,
                        stop_loss=stop_loss,
                        is_qualified=is_qualified,
                        filter_progression={
                            'date': datetime.now().isoformat(),
                            'filters': filters_passed,
                            'score': score
                        }
                    )
                    additions += 1
                    
                    # Create alert
                    if self.config.ALERT_ON_FILTER_IMPROVEMENT:
                        self.db.create_alert(
                            symbol=symbol,
                            alert_type='WATCHLIST_ADDITION',
                            severity='INFO',
                            message=f"{symbol} added to watchlist with {filters_passed} filters passed",
                            details={'filters': filters_passed, 'score': score}
                        )
                        alerts_created += 1
                
                else:
                    # Update existing
                    old_filters = existing['Filters_Passed']
                    
                    # Check for improvement
                    if filters_passed > old_filters:
                        if self.config.ALERT_ON_FILTER_IMPROVEMENT:
                            self.db.create_alert(
                                symbol=symbol,
                                alert_type='FILTER_IMPROVEMENT',
                                severity='WARNING',
                                message=f"{symbol} improved from {old_filters} to {filters_passed} filters",
                                details={'old_filters': old_filters, 'new_filters': filters_passed}
                            )
                            alerts_created += 1
                    
                    # Check for qualification
                    if is_qualified and not existing['Is_Qualified']:
                        if self.config.ALERT_ON_QUALIFICATION:
                            self.db.create_alert(
                                symbol=symbol,
                                alert_type='QUALIFICATION',
                                severity='CRITICAL',
                                message=f"{symbol} is now QUALIFIED for trading!",
                                details={'score': score, 'filters': filters_passed}
                            )
                            alerts_created += 1
                    
                    # Check for deterioration
                    if filters_passed < old_filters - self.config.DETERIORATION_THRESHOLD:
                        if self.config.ALERT_ON_DETERIORATION:
                            self.db.create_alert(
                                symbol=symbol,
                                alert_type='DETERIORATION',
                                severity='WARNING',
                                message=f"{symbol} deteriorated from {old_filters} to {filters_passed} filters",
                                details={'old_filters': old_filters, 'new_filters': filters_passed}
                            )
                            alerts_created += 1
                    
                    # Update entry
                    self.db.update_watchlist(
                        symbol=symbol,
                        filters_passed=filters_passed,
                        score=score,
                        price=price,
                        stop_loss=stop_loss,
                        is_qualified=is_qualified,
                        filter_progression={
                            'date': datetime.now().isoformat(),
                            'filters': filters_passed,
                            'score': score,
                            'change': filters_passed - old_filters
                        }
                    )
                    updates += 1
        
        logger.info(f"Watchlist updated: {additions} added, {updates} updated, {alerts_created} alerts created")
        
        return {
            'additions': additions,
            'updates': updates,
            'alerts': alerts_created
        }
    
    def _get_watchlist_entry(self, symbol: str) -> Optional[Dict]:
        """Get watchlist entry for a symbol"""
        df = self.db.get_active_watchlist()
        if df.empty:
            return None
        
        entry = df[df['Symbol'] == symbol]
        if entry.empty:
            return None
        
        return entry.iloc[0].to_dict()
    
    def monitor_watchlist_stocks(self):
        """Monitor all watchlist stocks for changes"""
        logger.info("Monitoring watchlist stocks...")
        
        watchlist_df = self.db.get_active_watchlist()
        if watchlist_df.empty:
            logger.info("No stocks in watchlist")
            return
        
        symbols = watchlist_df['Symbol'].tolist()
        tickers = [f"{symbol}.NS" for symbol in symbols]
        
        # Scan watchlist stocks
        results = []
        for ticker in tickers:
            result = self.scanner.scan_stock(ticker)
            if result:
                results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            self.update_watchlist_from_scan(results_df)
        
        logger.info("Watchlist monitoring complete")
    
    def cleanup_old_entries(self):
        """Remove old qualified or deteriorated stocks"""
        # This would be implemented in the database layer
        # For now, we'll log it
        logger.info("Cleanup of old watchlist entries would be performed here")


def main():
    """Main execution function"""
    # Initialize database
    db = DatabaseManager()
    
    # Initialize watchlist monitor
    config = WatchlistConfig()
    monitor = WatchlistMonitor(db, config)
    
    # Monitor watchlist
    monitor.monitor_watchlist_stocks()
    
    logger.info("Watchlist monitoring completed successfully!")


if __name__ == "__main__":
    main()
