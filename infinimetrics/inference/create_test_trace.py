import csv
import random
from pathlib import Path

def create_test_trace(output_file: str = "test_trace.csv", num_requests: int = 50):
    """Create a test trace file"""
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['request_id', 'arrival_timestamp_ms', 'input_token_num', 'output_token_num'])
        
        current_time = 0
        
        for i in range(num_requests):
            request_id = f"req-{i:04d}"
            
            # Random arrival interval (0–100ms)
            interval = random.uniform(0, 100)
            current_time += interval
            
            # Random token counts
            input_tokens = random.choice([64, 128, 256, 512])
            output_tokens = random.choice([64, 128, 256])
            
            writer.writerow([request_id, round(current_time, 2), input_tokens, output_tokens])
    
    print(f"Created test trace file: {output_file} with {num_requests} requests")

if __name__ == "__main__":
    create_test_trace()
