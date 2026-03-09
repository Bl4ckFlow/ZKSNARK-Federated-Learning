# simulation.py
import asyncio
import logging
import random
import hashlib
import numpy as np
import os
from typing import Dict

# Import your modules
from nodes.node import Node
from nodes.fl_types import UpdateType
from nodes.network_layer import NetworkLayer
from utils.serialization import serialize_model
from utils.zk_proof import generate_zk_proof

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("SIM")

class MockNetworkHub:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.delay_range = (0.01, 0.05) 

    def register_node(self, node: Node):
        self.nodes[node.node_id] = node

    async def send(self, sender_id: str, target_ip_port: tuple, message: dict):
        await asyncio.sleep(random.uniform(*self.delay_range))
        target_port = target_ip_port[1]
        target_node_id = f"Node_{target_port - 20000}"
        target_node = self.nodes.get(target_node_id)
        if target_node:
            await target_node.on_receive_message(message)


class MockNetworkLayer(NetworkLayer):
    def __init__(self, node_id: str, peers: list, hub: MockNetworkHub):
        self.node_id = node_id
        self.peers = peers
        self.hub = hub
        self.max_retry_attempts = 0
        self.retry_delay = 0

    async def broadcast_update(self, update: UpdateType) -> dict:
        msg = {"type": "update", "data": update.to_dict()}
        for peer in self.peers:
            asyncio.create_task(self.hub.send(self.node_id, peer, msg))
        return {}

    async def broadcast_vote(self, vote) -> dict:
        msg = {"type": "vote", "data": vote.to_dict()}
        for peer in self.peers:
            asyncio.create_task(self.hub.send(self.node_id, peer, msg))
        return {}
    
    async def gossip_loop(self, *args, **kwargs): pass

def create_valid_update(height: int, prev_model: dict, node_id: str) -> UpdateType:
    prev_weights = np.array(prev_model["weights"])
    prev_bias = prev_model["bias"]
    
    grad_w = np.random.uniform(-0.1, 0.1, size=prev_weights.shape)
    grad_b = np.random.uniform(-0.1, 0.1)
    lr = 0.01
    
    new_weights = prev_weights - (lr * grad_w)
    new_bias = prev_bias - (lr * grad_b)
    
    new_params = {"weights": new_weights.tolist(), "bias": new_bias}

    serialized = serialize_model(new_params)
    params_hash = hashlib.sha256(serialized).hexdigest()
    prev_hash = hashlib.sha256(serialize_model(prev_model)).hexdigest()
    
    logger.info(f"[{node_id}] Generating ZK Proof for H={height}...")
    zk_proof = generate_zk_proof(
        prev_model_hash=prev_hash,
        serialized_new_params=serialized,
        prev_model=prev_model,
        new_model=new_params,
        learning_rate=lr,
        enable_strict_verification=True 
    )

    payload = f"{prev_hash}|{params_hash}|{height}".encode()
    u_id = hashlib.sha256(payload).hexdigest()

    return UpdateType(
        prev_model_hash=prev_hash,
        params_hash=params_hash,
        new_params=new_params,
        model_height=height,
        zk_proof=zk_proof,
        sender_pubkey=None,
        signature=None,
        update_id=u_id
    )

async def run_simulation():
    print("\n" + "="*50)
    logger.info(">>> STARTING ZK-ENABLED FEDERATED LEARNING SIM <<<")
    print("="*50 + "\n")

    hub = MockNetworkHub()
    nodes = []
    peers_config = {
        "Node_1": [("localhost", 20002), ("localhost", 20003)],
        "Node_2": [("localhost", 20001), ("localhost", 20003)],
        "Node_3": [("localhost", 20001), ("localhost", 20002)],
    }
    genesis_model = {"weights": [0.0, 0.0], "bias": 0.0}

    for i in range(1, 4):
        node_id = f"Node_{i}"
        node = Node(node_id, peers_config[node_id], voting_window=2.0, min_votes=2, enable_strict_zk=True, learning_rate=0.01)
        node.network_layer = MockNetworkLayer(node_id, peers_config[node_id], hub)
        node.set_model(genesis_model)
        nodes.append(node)
        hub.register_node(node)

    logger.info("Nodes initialized. Starting Test Sequence...")

    # TEST 1
    logger.info("\n--- STEP 1: Broadcasting H=1 (Happy Path) ---")
    u1 = create_valid_update(1, genesis_model, "Node_1")
    await asyncio.gather(nodes[0].on_receive_update(u1), nodes[1].on_receive_update(u1), nodes[2].on_receive_update(u1))
    await asyncio.sleep(4.0) 
    for n in nodes: assert n.model_height == 1, f"{n.node_id} failed to reach H=1"
    logger.info("All nodes verified ZK proof and reached H=1")

    # TEST 2
    logger.info("\n--- STEP 2: Partitioning Node 3 (Simulating Gap) ---")
    u2 = create_valid_update(2, u1.new_params, "Node_2")
    await nodes[0].on_receive_update(u2)
    await nodes[1].on_receive_update(u2)
    await asyncio.sleep(4.0)
    assert nodes[0].model_height == 2
    assert nodes[2].model_height == 1 
    logger.info("Partial consensus: N1 & N2 are at H=2. N3 is left behind at H=1.")

    # TEST 3
    logger.info("\n--- STEP 3: Sending H=3 to ALL (Node 3 receives Future Update) ---")
    u3 = create_valid_update(3, u2.new_params, "Node_1")
    await nodes[0].on_receive_update(u3)
    await nodes[1].on_receive_update(u3)
    await nodes[2].on_receive_update(u3) 
    await asyncio.sleep(4.0)
    assert nodes[2].model_height == 1
    logger.info("Node 3 correctly buffered H=3 (waiting for H=2)")

    # TEST 4
    logger.info("\n--- STEP 4: Providing Missing Link (H=2) to Node 3 ---")
    await nodes[2].on_receive_update(u2)
    logger.info("⏳ Waiting for Chain Reaction...")
    await asyncio.sleep(6.0) 
    assert nodes[2].model_height == 3
    logger.info("CHAIN REACTION SUCCESS! Node 3 caught up from H=1 -> H=3 automatically.")
    print("\n" + "="*50)
    logger.info(">>> SIMULATION COMPLETED SUCCESSFULLY <<<")
    print("="*50 + "\n")

if __name__ == "__main__":
    if not os.path.exists("./circuits/update_check_final.zkey"):
        logger.error("❌ ZK Keys not found! Run setup_zk.py")
    else:
        asyncio.run(run_simulation())