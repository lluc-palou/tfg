import os
import re
import time
import json
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from websocket import create_connection, WebSocketConnectionClosedException

# BitMEX LOB WebSocket endpoint
LOB_WS_URL = "wss://www.bitmex.com/realtime?subscribe=orderBookL2:XBTUSD"

# LOB snapshot storage
lob_snapshot = None
lock = threading.Lock()

# Output directory
LOB_OUTPUT_DIR = "lob_data"
os.makedirs(LOB_OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    """
    Returns current timestamp in ISO format.
    """
    return datetime.now(timezone.utc).isoformat()

def get_lob_output_file():
    """
    Returns current day LOB filepath.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(LOB_OUTPUT_DIR, f"{date_str}.parquet")

def listen_lob():
    """
    Connects to BitMEX LOB WebSocket and maintains current order book snapshot.
    """
    global lob_snapshot
    order_book = {}

    while True:
        try:
            ws = create_connection(LOB_WS_URL)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to BitMEX LOB WebSocket")

            while True:
                data = json.loads(ws.recv())

                if data.get("table") == "orderBookL2":
                    with lock:
                        action = data["action"]

                        if action == "partial":
                            # Full snapshot - rebuild entire order book
                            order_book.clear()
                            for entry in data["data"]:
                                order_book[entry["id"]] = {
                                    "price": entry["price"],
                                    "size": entry["size"],
                                    "side": entry["side"]
                                }
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] LOB snapshot initialized with {len(order_book)} levels")

                        else:
                            # Incremental updates
                            for entry in data["data"]:
                                order_id = entry["id"]

                                if action == "insert":
                                    order_book[order_id] = {
                                        "price": entry["price"],
                                        "size": entry["size"],
                                        "side": entry["side"]
                                    }

                                elif action == "update":
                                    if order_id in order_book:
                                        order_book[order_id]["size"] = entry["size"]

                                elif action == "delete":
                                    order_book.pop(order_id, None)

                        # Rebuild sorted bids/asks arrays
                        bids = []
                        asks = []

                        for order in order_book.values():
                            if order["side"] == "Buy":
                                bids.append([order["price"], order["size"]])
                            else:
                                asks.append([order["price"], order["size"]])

                        # Sort price levels: bids descending, asks ascending
                        bids.sort(key=lambda x: -x[0])
                        asks.sort(key=lambda x: x[0])

                        lob_snapshot = {"bids": bids, "asks": asks}

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LOB WebSocket error: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)

def generate_backup_filepath(filepath):
    """
    Generate backup filepath with incremental numbering.
    """
    base, ext = os.path.splitext(filepath)
    pattern = re.compile(re.escape(base) + r'_(\d+)' + re.escape(ext))
    
    # Find existing backup numbers
    existing_backups = [
        int(m.group(1)) for f in os.listdir(os.path.dirname(filepath))
        if (m := pattern.fullmatch(f))
    ]

    next_index = max(existing_backups, default=0) + 1
    return f"{base}_{next_index}{ext}"

def append_to_parquet(filepath, df):
    """
    Append DataFrame to Parquet file, handling corruption gracefully.
    """
    table = pa.Table.from_pandas(df)

    if not os.path.exists(filepath):
        pq.write_table(table, filepath)
        return
    
    try:
        existing_table = pq.read_table(filepath)
        combined = pa.concat_tables([existing_table, table])
        pq.write_table(combined, filepath)
    
    except (pa.lib.ArrowInvalid, OSError) as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Corrupted Parquet file at {filepath}. Reason: {e}")
        backup_path = generate_backup_filepath(filepath)
        os.rename(filepath, backup_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Corrupted file moved to: {backup_path}")
        pq.write_table(table, filepath)

def record_snapshots():
    """
    Records LOB snapshots every 30 seconds to Parquet files.
    """
    snapshot_count = 0
    
    while True:
        # Wait 30 seconds before taking snapshot
        time.sleep(30)
        
        with lock:
            if lob_snapshot:
                # Create LOB snapshot record
                timestamp = get_timestamp()
                
                lob_df = pd.DataFrame([{
                    "timestamp": timestamp,
                    "bids": json.dumps(lob_snapshot["bids"]),
                    "asks": json.dumps(lob_snapshot["asks"])
                }])

                # Save to Parquet file
                lob_file = get_lob_output_file()
                append_to_parquet(lob_file, lob_df)
                
                snapshot_count += 1
                
                # Calculate bid-ask spread for logging
                if lob_snapshot["bids"] and lob_snapshot["asks"]:
                    best_bid = lob_snapshot["bids"][0][0]
                    best_ask = lob_snapshot["asks"][0][0]
                    spread = best_ask - best_bid
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot #{snapshot_count} saved. "
                          f"Bid: {best_bid}, Ask: {best_ask}, Spread: {spread:.2f}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot #{snapshot_count} saved (empty book)")
            
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No LOB snapshot available yet, waiting...")

def start_collection():
    """
    Initialize and start LOB collection threads.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting LOB data collection...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output directory: {os.path.abspath(LOB_OUTPUT_DIR)}")
    
    # Start LOB WebSocket listener
    threading.Thread(target=listen_lob, daemon=True).start()
    
    # Start snapshot recorder
    threading.Thread(target=record_snapshots, daemon=True).start()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection threads started. Taking snapshots every 30 seconds.")

if __name__ == "__main__":
    # Start the LOB collection system
    start_collection()

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LOB collection running. Press Ctrl+C to stop.")
        while True:
            # Keep main thread alive and show status every 10 minutes
            time.sleep(600)  # 10 minutes
            
            # Show current file info
            current_file = get_lob_output_file()
            if os.path.exists(current_file):
                file_size = os.path.getsize(current_file)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {current_file} - {file_size/1024/1024:.2f}MB")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: No data file created yet")

    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] LOB data collection stopped by user")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Collection stopped due to error: {e}")