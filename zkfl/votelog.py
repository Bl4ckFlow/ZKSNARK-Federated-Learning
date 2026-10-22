"""Write-ahead log for this node's own votes.

Both cases of the agreement proof rest on ``a correct node signs one vote per
slot``.  The fault model allows a correct node to crash and restart, and a node
that crashed after broadcasting a prevote and restarted without it would sign a
*second* vote in the same slot -- self-equivocation.  Its peers would blacklist
it on transferable evidence, correctly, and the proof's hypothesis would simply
be false: the node is not Byzantine, but it behaves like one.

So a vote is durable before it is sent, never after.  On restart the log is
replayed and a slot that is already occupied re-sends the vote it recorded
rather than signing a new one, which makes the send idempotent: a peer may
receive the same signed vote twice, and duplicate delivery is already the
common case under gossip.

``InMemoryVoteLog`` is the default because most runs are simulations that never
restart, and it says plainly that it is not crash-safe rather than pretending.
``DurableVoteLog`` is what a deployment uses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Protocol

from zkfl.messages import Vote

#: (height, sub_round, phase) -- the slot a vote occupies.  The voter is
#: implicit: this log holds only our own votes.
Slot = tuple[int, int, str]


def slot_of(vote: Vote) -> Slot:
    return (vote.height, vote.sub_round, vote.phase)


class VoteLog(Protocol):
    #: whether a restart preserves what was recorded
    durable: bool

    def recall(self, slot: Slot) -> Optional[Vote]: ...
    def record(self, vote: Vote) -> None: ...
    def close(self) -> None: ...


class InMemoryVoteLog:
    """Enough to enforce one-vote-per-slot within a single process.

    Not crash-safe, and deliberately not pretending to be: a restart loses the
    log and the next vote in a live slot would be a second one.  Simulations
    never restart, so this is the honest default for them.
    """

    durable = False

    def __init__(self) -> None:
        self._votes: dict[Slot, Vote] = {}

    def recall(self, slot: Slot) -> Optional[Vote]:
        return self._votes.get(slot)

    def record(self, vote: Vote) -> None:
        self._votes.setdefault(slot_of(vote), vote)

    def close(self) -> None:
        """Nothing to flush -- present so the two logs are interchangeable."""

    def __len__(self) -> int:
        return len(self._votes)


class DurableVoteLog:
    """Append, flush, ``fsync`` -- then the caller may send.

    The ordering is the whole point.  ``record`` returns only once the bytes are
    on the device, so a crash between recording and broadcasting loses the
    broadcast (harmless: peers time out and the next sub-round proceeds) rather
    than losing the record (not harmless: the node would sign a second vote in
    the slot).
    """

    durable = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._votes: dict[Slot, Vote] = {}
        self._replay()
        self._fh = open(self.path, "a", encoding="utf-8")

    def _replay(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    vote = Vote.from_dict(json.loads(line))
                except (ValueError, TypeError, json.JSONDecodeError):
                    # A torn final record is expected after a crash mid-write.
                    # Everything before it is intact, which is what append-only
                    # buys; stop rather than guess at the remainder.
                    break
                self._votes.setdefault(slot_of(vote), vote)

    def recall(self, slot: Slot) -> Optional[Vote]:
        return self._votes.get(slot)

    def record(self, vote: Vote) -> None:
        slot = slot_of(vote)
        if slot in self._votes:
            return
        self._fh.write(json.dumps(vote.to_dict(), sort_keys=True,
                                  separators=(",", ":")) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._votes[slot] = vote

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __len__(self) -> int:
        return len(self._votes)
