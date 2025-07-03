import os
import time
import json
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from websocket import create_connection, WebSocketConnectionClosedException

# BitMEX WS (WebSocket) endpoints
LOB_WS_URL = "wss://www.bitmex.com/realtime?subscribe=orderBookL2:XBTUSD"
TRADE_WS_URL = "wss://www.bitmex.com/realtime?subscribe=trade:XBTUSD"

# Buffers to store incoming trades and order book updates
trades_buffer = []
lob_snapshot = None
lock = threading.Lock()

# Output directories
LOB_OUTPUT_DIR = "lob_data"
TRADE_OUTPUT_DIR = "trade_data"
os.makedirs(LOB_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRADE_OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    """
    Returns current day timestamp.
    """
    return datetime.now(timezone.utc).isoformat()

def get_lob_output_file():
    """
    Returns current day LOB filepath.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(LOB_OUTPUT_DIR, f"{date_str}.parquet")

def get_trade_output_file():
    """
    Returns current day trade summary filepath.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(TRADE_OUTPUT_DIR, f"{date_str}.parquet")

def listen_trades():
    """
    Connects to BitMEX trade WS and collects incoming live trade events.
    """
    while True:
        try:
            ws = create_connection(TRADE_WS_URL)

            while True:
                data = json.loads(ws.recv())

                if data.get("table") == "trade":
                    with lock:
                        for trade in data["data"]:
                            trades_buffer.append(
                                {
                                    "timestamp": trade["timestamp"],
                                    "price": trade["price"],
                                    "size": trade["size"]
                                }
                            )

        except Exception as e:
            print(f"Trade WebSocket error: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)

def listen_lob():
    """
    Connects to BitMEX LOB WS and updates latest emitted snapshot.
    """
    global lob_snapshot

    while True:
        try:
            ws = create_connection(LOB_WS_URL)

            while True:
                data = json.loads(ws.recv())

                if data.get("table") == "orderBookL2":
                    with lock:
                        if data["action"] == "partial":
                            bids = []
                            asks = []

                            for entry in data["data"]:
                                if entry["side"] == "Buy":
                                    bids.append([entry["price"], entry["size"]])

                                else:
                                    asks.append([entry["price"], entry["size"]])

                            lob_snapshot = {"bids": bids, "asks": asks}

        except Exception as e:
            print(f"LOB WebSocket error: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)

def append_to_parquet(filepath, df):
    """
    Given current filepath saves both LOB snapshot and trade summary outputs using Parquet file format.
    """
    table = pa.Table.from_pandas(df)

    if not os.path.exists(filepath):
        pq.write_table(table, filepath)

    else:
        existing_table = pq.read_table(filepath)
        combined = pa.concat_tables([existing_table, table])
        pq.write_table(combined, filepath)

def record_snapshot():
    """
    Saves the current LOB snapshot and trade summary separately.
    """
    while True:
        # LOB snapshot sampling period
        time.sleep(30)

        with lock:
            if lob_snapshot:
                # Computes total volume over sampling period
                now = datetime.now(timezone.utc)
                cutoff = now.timestamp() - 30
                recent_trades = [t for t in trades_buffer if datetime.strptime(t["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp() >= cutoff]

                if recent_trades:
                    last_price = recent_trades[-1]["price"]
                    volume = sum(t["size"] for t in recent_trades)

                else:
                    last_price = None
                    volume = 0.0

                # LOB snapshot
                lob_df = pd.DataFrame(
                    [
                        {   
                            "timestamp": get_timestamp(),
                            "bids": json.dumps(lob_snapshot["bids"]),
                            "asks": json.dumps(lob_snapshot["asks"])
                        }
                    ]
                )

                # Trade summary
                trade_df = pd.DataFrame(
                    [
                        {   
                            "timestamp": get_timestamp(),
                            "last_price": last_price,
                            "volume": volume
                        }
                    ]
                )

                # Obtains current file
                lob_file = get_lob_output_file()
                trade_file = get_trade_output_file()

                # Appends data to the corresponding file
                append_to_parquet(lob_file, lob_df)
                append_to_parquet(trade_file, trade_df)

                # Cleans up old sampling period trades
                trades_buffer.clear()

def start_recording():
    """
    Initializes and starts all working threads.
    """
    threading.Thread(target=listen_trades, daemon=True).start()
    threading.Thread(target=listen_lob, daemon=True).start()
    threading.Thread(target=record_snapshot, daemon=True).start()

if __name__ == "__main__":
    # Starts the data collection system
    start_recording()

    try:
        while True:
            # Prevents the script from exiting execution
            time.sleep(60)

    except KeyboardInterrupt:
        print("Data collection stopped")