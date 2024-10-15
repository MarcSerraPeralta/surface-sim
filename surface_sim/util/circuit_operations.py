import stim


def merge_circuits(*circuits: stim.Circuit) -> stim.Circuit:
    """Returns a circuit in which the given circuits have been merged
    following the TICK blocks.

    It assumes that the qubits are only involved in one operation between TICKs.

    Parameters
    ----------
    *circuits
        Circuits to merge.

    Returns
    -------
    merged_circuit
        Circuit from merging the given circuits.
    """
    if any([not isinstance(c, stim.Circuit) for c in circuits]):
        raise TypeError("The given circuits are not stim.Circuits.")
    if len(set(c.num_ticks for c in circuits)) != 1:
        raise ValueError("All the circuits must have the same number of TICKs.")

    # split circuits into TICK blocks
    blocks = [[] for _ in range(circuits[0].num_ticks + 1)]
    for circuit in circuits:
        block_id = 0
        for instr in circuit.flattened():
            if instr.name == "TICK":
                block_id += 1
            else:
                blocks[block_id].append(instr)

    # sort the instructions in each block for better readability in the circuit
    blocks = [sorted(block, key=lambda x: x.name) for block in blocks]

    # merge instructions in blocks and into a circuit.
    # remember that it is not possible to add stim.CircuitInstruction to stim.Circuit
    tick = stim.Circuit("TICK")
    merged_blocks = [
        stim.Circuit("\n".join(map(str, block))) + tick for block in blocks
    ]
    merged_blocks[-1].pop(-1)  # removes extra TICK at the end
    merged_circuit = sum(merged_blocks, start=stim.Circuit())

    return merged_circuit
