"""
Automated Scheduler for Trading Scanner
Runs scans at specified times
"""

import schedule
import time
from datetime import datetime
import logging

from scanner import main as run_scanner
from watchlist_monitor import main as run_watchlist_monitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def scheduled_full_scan():
    """Run full universe scan"""
    logger.info("=" * 60)
    logger.info("Starting scheduled FULL SCAN")
    logger.info("=" * 60)
    
    try:
        run_scanner()
        logger.info("✅ Full scan completed successfully")
    except Exception as e:
        logger.error(f"❌ Full scan failed: {e}")


def scheduled_watchlist_update():
    """Run watchlist update"""
    logger.info("=" * 60)
    logger.info("Starting scheduled WATCHLIST UPDATE")
    logger.info("=" * 60)
    
    try:
        run_watchlist_monitor()
        logger.info("✅ Watchlist update completed successfully")
    except Exception as e:
        logger.error(f"❌ Watchlist update failed: {e}")


def main():
    """Main scheduler function"""
    
    logger.info("Trading Scanner Scheduler Started")
    logger.info("=" * 60)
    
    # Schedule full scan daily at 6:30 PM IST (after market close at 3:30 PM + buffer)
    schedule.every().monday.at("18:30").do(scheduled_full_scan)
    schedule.every().tuesday.at("18:30").do(scheduled_full_scan)
    schedule.every().wednesday.at("18:30").do(scheduled_full_scan)
    schedule.every().thursday.at("18:30").do(scheduled_full_scan)
    schedule.every().friday.at("18:30").do(scheduled_full_scan)
    
    # Schedule watchlist update twice daily
    schedule.every().day.at("10:00").do(scheduled_watchlist_update)  # Morning
    schedule.every().day.at("16:00").do(scheduled_watchlist_update)  # After market
    
    logger.info("Schedule configured:")
    logger.info("  - Full Scan: Mon-Fri at 18:30")
    logger.info("  - Watchlist Update: Daily at 10:00 and 16:00")
    logger.info("=" * 60)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
