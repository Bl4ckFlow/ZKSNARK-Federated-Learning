import numpy as np
import asyncio
from nodes.node import Node

async def run_test():
    ip = "127.0.0.1"
    base_port = 20000
    num_nodes = 7
    
    genesis_model = {
        "weights": np.zeros(4).tolist(), 
        "bias": 0.0
    }

    bullshit_data = {
        "X": np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [5.8, 2.7, 4.1, 1.0], [6.0, 3.4, 4.5, 1.6]]),
        "y": np.array([0, 0, 1, 1])
    }

    # Helper function to generate Full Mesh peers
    # Every node gets the (IP, Port) of every OTHER node in the list
    def get_full_mesh_peers(my_id):
        return [(ip, base_port + j) for j in range(num_nodes) if j != my_id]

    # 1. Initialize Nodes with Full Mesh Connectivity
    nodes = []
    for i in range(num_nodes):
        n = Node(
            node_id=str(i), 
            peers=get_full_mesh_peers(i),
            min_votes=4,  # Majority for 7 nodes is 4 (simple BFT)
            voting_window=5.0
        )
        n.set_model(genesis_model)
        nodes.append(n)

    # Wrap runs in tasks to manage them
    server_tasks = [asyncio.create_task(n.run(base_port + i)) for i, n in enumerate(nodes)]

    async def monitor_and_trigger():
        try:
            await asyncio.sleep(3)
            print("\n" + "="*50)
            print("[!] DATA INJECTION: Nodes 1 and 5")
            print("="*50 + "\n")
            
            nodes[1].set_local_data(bullshit_data)
            nodes[5].set_local_data(bullshit_data)

            print("[*] Waiting for consensus...")
            while not all(n.model_height >= 1 for n in nodes):
                await asyncio.sleep(1)

            print("\n" + "X"*50)
            print("VOTING AUDIT TRAIL (Who voted for whom)")
            print("X"*50 + "\n")

            for n in nodes:
                # Access the private voting record for Height 1
                # Format: {voter_id: VoteMessage}
                votes = n._votes_by_height.get(1, [])
                print(f"--- Node {n.node_id} Audit ---")
                if not votes:
                    print("  [!] No local vote records found (already cleared or missed)")
                for vote in votes:
                    # 'voted_update_id' tells you WHICH update they picked
                    print(f"  Voter: {vote.voter_node_id} -> Choice: {vote.voted_update_id[:8]}")
            
            print("\n" + "X"*50)
            print("FINAL CONSENSUS REACHED")
            print("X"*50 + "\n")

            for n in nodes:
                m = n.get_model()
                if m:
                    w = np.array(m['weights']) 
                    print(f"Node {n.node_id} | H: {n.model_height} | Hash: {n.local_model_hash[:8]} | Weights: {w}")

        finally:
            for t in server_tasks:
                t.cancel()

    await asyncio.gather(*server_tasks, monitor_and_trigger(), return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n[!] Simulation stopped.")