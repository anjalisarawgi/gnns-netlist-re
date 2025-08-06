#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from typing import Dict, List, Tuple, Optional


class CircuitNode:
    def __init__(self,
                 name: str,
                 bool_func: str,
                 inputs: List[str],
                 outputs: List[str],
                 processed: str,
                 count: int) -> None:
        self._name = name
        self._bool_func = bool_func
        self._inputs = list(inputs)  
        self._outputs = list(outputs)  # signals driven by this gate
        # list of boolean function types for gates this node feeds
        self._fwdgates: List[Optional[str]] = []
        # list of instance names for gates this node feeds
        self._fwdgates_inst: List[Optional[str]] = []
        # list of boolean function types for gates feeding this node
        self._fedbygates: List[Optional[str]] = []
        # list of instance names for gates feeding this node
        self._fedbygates_inst: List[Optional[str]] = []
        self._processed = processed  # module type or top module id
        self._count = count  # numeric identifier assigned globally

    # Getter and setter methods mirror those in the Perl version.
    def get_name(self) -> str:
        return self._name

    def get_bool_func(self) -> str:
        return self._bool_func

    def get_inputs(self) -> List[str]:
        return list(self._inputs)

    def get_outputs(self) -> List[str]:
        return list(self._outputs)

    def get_fwdgates(self) -> List[Optional[str]]:
        return list(self._fwdgates)

    def set_fwdgates(self, gates: List[Optional[str]]) -> None:
        self._fwdgates = list(gates)

    def get_fwdgates_inst(self) -> List[Optional[str]]:
        return list(self._fwdgates_inst)

    def set_fwdgates_inst(self, inst: List[Optional[str]]) -> None:
        self._fwdgates_inst = list(inst)

    def get_fedbygates(self) -> List[Optional[str]]:
        return list(self._fedbygates)

    def set_fedbygates(self, gates: List[Optional[str]]) -> None:
        self._fedbygates = list(gates)

    def get_fedbygates_inst(self) -> List[Optional[str]]:
        return list(self._fedbygates_inst)

    def set_fedbygates_inst(self, inst: List[Optional[str]]) -> None:
        self._fedbygates_inst = list(inst)

    def get_processed(self) -> str:
        return self._processed

    def get_count(self) -> int:
        return self._count


def expand_buses(signals: List[str]) -> List[str]:
    """Expand bus declarations like ``[7:0] data`` into individual bits.

    The Perl code processes netlist inputs and outputs that are given
    as buses (e.g. ``[3:0] in``) by generating individual wires
    ``in[0]``, ``in[1]`` and so on.  This helper function takes a
    list of raw signal declarations and returns the flattened list of
    signals after expansion.
    """
    expanded: List[str] = []
    bus_pattern = re.compile(r"\s*\[(\d+)\:(\d+)\]\s+(\S+)")
    for item in signals:
        item = item.strip()
        m = bus_pattern.match(item)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            name = m.group(3)
            # handle reversed ranges
            idx_start, idx_end = min(start, end), max(start, end)
            for i in range(idx_start, idx_end + 1):
                expanded.append(f"{name}[{i}]")
        else:
            expanded.append(item)
    return expanded


def parse_module_ios(lines: List[str]) -> Tuple[List[str], List[str], str]:
    """Extract top module input and output signals from a list of lines.

    The Perl script repeatedly resets the ``Module_Inputs`` and
    ``Module_Outputs`` arrays for each ``module``/``endmodule`` pair
    and copies them into ``Netlist_Inputs`` and ``Netlist_Outputs``
    upon encountering an ``endmodule`` keyword.  This helper
    replicates that logic: it returns the final lists of inputs and
    outputs corresponding to the last module encountered as well as
    the name of that module.

    Parameters
    ----------
    lines : List[str]
        The contents of a Verilog file split into individual lines.

    Returns
    -------
    tuple of (inputs, outputs, module_name)
        ``inputs`` and ``outputs`` contain the flattened signal names;
        ``module_name`` is the name of the top module.
    """
    module_inputs: List[str] = []
    module_outputs: List[str] = []
    netlist_inputs: List[str] = []
    netlist_outputs: List[str] = []
    module_id: str = ""
    top_module: str = ""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Capture module declaration
        m = re.match(r"^\s*module\s+(\w+)\b", line)
        if m:
            module_id = m.group(1)
            top_module = module_id
            i += 1
            continue
        # Capture endmodule and snapshot IOs for this module
        if re.match(r"^\s*endmodule\b", line):
            netlist_inputs = module_inputs
            netlist_outputs = module_outputs
            module_inputs = []
            module_outputs = []
            i += 1
            continue
        # Handle input declarations
        m_in = re.match(r"^\s*input\s+.*", line)
        if m_in:
            # Remove the leading keyword
            working = line
            # If declaration ends on this line (with semicolon), process it
            if re.search(r";\s*$", working):
                working = re.sub(r"^\s*input\s+", "", working)
                working = working.replace(";", "")
                # Remove newlines and trim
                working = working.strip()
                if working:
                    found_inputs = [x.strip() for x in working.split(',') if x.strip()]
                    module_inputs.extend(found_inputs)
                i += 1
                continue
            # Otherwise the declaration spans multiple lines
            working = re.sub(r"^\s*input\s+", "", working)
            working = working.replace(";", "")
            working = working.strip()
            if working:
                found_inputs = [x.strip() for x in re.split(r",\s*", working) if x.strip()]
                module_inputs.extend(found_inputs)
            # Read subsequent lines until encountering a semicolon
            i += 1
            while i < n:
                next_line = lines[i]
                if re.search(r";\s*$", next_line):
                    cleaned = next_line.replace(";", "").strip()
                    if cleaned:
                        found_inputs = [x.strip() for x in re.split(r",\s*", cleaned) if x.strip()]
                        module_inputs.extend(found_inputs)
                    break
                cleaned = next_line.strip()
                if cleaned:
                    found_inputs = [x.strip() for x in re.split(r",\s*", cleaned) if x.strip()]
                    module_inputs.extend(found_inputs)
                i += 1
            i += 1
            continue
        # Handle output declarations (similar to inputs)
        m_out = re.match(r"^\s*output\s+.*", line)
        if m_out:
            working = line
            if re.search(r";\s*$", working):
                working = re.sub(r"^\s*output\s+", "", working)
                working = working.replace(";", "")
                working = working.strip()
                if working:
                    found_outputs = [x.strip() for x in working.split(',') if x.strip()]
                    module_outputs.extend(found_outputs)
                i += 1
                continue
            working = re.sub(r"^\s*output\s+", "", working)
            working = working.replace(";", "")
            working = working.strip()
            if working:
                found_outputs = [x.strip() for x in re.split(r",\s*", working) if x.strip()]
                module_outputs.extend(found_outputs)
            i += 1
            while i < n:
                next_line = lines[i]
                if re.search(r";\s*$", next_line):
                    cleaned = next_line.replace(";", "").strip()
                    if cleaned:
                        found_outputs = [x.strip() for x in re.split(r",\s*", cleaned) if x.strip()]
                        module_outputs.extend(found_outputs)
                    break
                cleaned = next_line.strip()
                if cleaned:
                    found_outputs = [x.strip() for x in re.split(r",\s*", cleaned) if x.strip()]
                    module_outputs.extend(found_outputs)
                i += 1
            i += 1
            continue
        i += 1
    # Expand buses in the final input/output lists
    netlist_inputs = expand_buses(netlist_inputs)
    netlist_outputs = expand_buses(netlist_outputs)
    return netlist_inputs, netlist_outputs, top_module


def parse_netlist_file(
    file_path: str,
    ml_count_start: int,
    assign_count_start: int,
    trial: int,
    top_module_name: str,
) -> Tuple[
    Dict[str, CircuitNode],  # local circuit mapping
    List[str],               # local list of gates
    List[int],               # local training indices
    List[int],               # local validation indices
    List[int],               # local test indices
    set,                     # netlist input signals for this file
    int,                     # updated ml_count
    int                      # updated assign_count
]:
    """Parse a single Verilog file and return its circuit representation.

    Unlike the Perl version which reused global data structures
    across files, this helper returns all relevant per‑file data so
    that the caller can process each netlist in isolation.  The
    returned ``ml_count`` and ``assign_count`` values should be
    threaded through subsequent calls to ensure unique identifiers
    across files.
    """
    ml_count = ml_count_start
    assign_count = assign_count_start
    # Read the entire file into memory to facilitate multi‑line parsing
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Obtain flattened lists of input and output signals for the top module
    netlist_inputs, netlist_outputs, top_module = parse_module_ios(lines)
    # If a module name was detected in the file, prefer it over the
    # externally supplied ``top_module_name``.  Otherwise use the
    # value passed in from the caller.
    if top_module:
        top_module_name = top_module
    # Build quick membership sets for inputs and outputs to test PI/PO
    netlist_inputs_set = set(netlist_inputs)
    netlist_outputs_set = set(netlist_outputs)
    # Local structures
    local_circuit: Dict[str, CircuitNode] = {}
    local_list_of_gates: List[str] = []
    local_tr: List[int] = []
    local_va: List[int] = []
    local_te: List[int] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        # Skip empty lines early
        if not stripped:
            i += 1
            continue
        # Parse assignment buffer: assign <out> = <in>;
        assign_match = re.match(r"^\s*assign\s+(\S+)\s*=\s*(\S+)\s*;", line)
        if assign_match:
            out_signal = assign_match.group(1)
            in_signal = assign_match.group(2)
            node_name = f"assign_{assign_count}"
            # Determine dataset split membership
            if trial == 3:
                local_tr.append(ml_count)
            elif trial == 1:
                local_va.append(ml_count)
            else:
                local_te.append(ml_count)
            # Create a new node for this buffer
            node = CircuitNode(
                name=node_name,
                bool_func="BUF",
                inputs=[in_signal],
                outputs=[out_signal],
                processed=top_module_name,
                count=ml_count,
            )
            # If this assignment drives a primary output, mark its
            # forward connection as PO
            if out_signal in netlist_outputs_set:
                node.set_fwdgates(["PO"])
                node.set_fwdgates_inst([out_signal])
            local_list_of_gates.append(node_name)
            local_circuit[node_name] = node
            ml_count += 1
            assign_count += 1
            i += 1
            continue
        # Skip module keywords to avoid false positives when matching cells
        if re.match(r"^\s*module\b", stripped) or re.match(r"^\s*endmodule\b", stripped):
            i += 1
            continue
        # Cell definitions may span multiple lines until a semicolon is
        # encountered.  Concatenate such lines before processing.
        combined_line = line.rstrip('\n')
        if '(' in combined_line and ';' not in combined_line:
            j = i + 1
            while j < n:
                next_line = lines[j].strip()
                combined_line += next_line
                if ';' in next_line:
                    break
                j += 1
            i = j  # Jump to the line containing the semicolon
        # Attempt to match a cell definition on the (potentially
        # combined) line.  We expect the pattern ``CellName InstanceName
        # ( .Port(Signal), ... );``
        cell_match = re.match(
            r"^\s*(\S+)\s+(\S+)\s*\(\s*(.+)\s*\)\s*;\s*$",
            combined_line)
        if cell_match:
            cell_name = cell_match.group(1)
            instance_name = cell_match.group(2)
            ports_str = cell_match.group(3)
            # Split ports on commas, but ignore empty entries
            raw_ports = [p.strip() for p in ports_str.split(',') if p.strip()]
            ports: Dict[str, str] = {}
            for entry in raw_ports:
                pm = re.match(r"\.([A-Za-z0-9_]+)\(\s*(\S+)\s*\)", entry)
                if pm:
                    port_name = pm.group(1)
                    signal = pm.group(2)
                    ports[port_name] = signal
            # Derive the boolean function by stripping suffixes and digits
            bool_fun = cell_name
            # Remove anything following the first underscore
            bool_fun = bool_fun.split('_')[0]
            # Remove trailing digits and any trailing non‑digit characters
            bool_fun = re.sub(r"\d+\D*$", "", bool_fun)
            # Determine the list of input signals for this gate.  The Perl
            # code checks for a long list of port names in a specific
            # order; we mirror that ordering here.  Only ports present
            # in ``ports`` are appended to ``inputs``.
            inputs: List[str] = []
            for key in [
                "A", "D", "B", "A1N", "B0N", "B1N", "A0N", "E", "F",
                "S0", "S1", "CN", "BN", "AN", "DN", "C", "C1", "C2",
                "CI", "B3", "B2", "B1", "B0", "B0N", "C0", "S",
                "A3", "A0", "A4", "A1", "A2"
            ]:
                if key in ports:
                    inputs.append(ports[key])
            # Determine the output signals of the gate following the
            # conditional logic from the Perl script.
            outputs: List[str] = []
            if "D" in ports and "Q" in ports:
                outputs.append(ports["Q"])
            elif "CO" in ports and "S" in ports:
                outputs.append(ports["S"])
                outputs.append(ports["CO"])
            elif "CO" in ports:
                outputs.append(ports["CO"])
            elif "D" in ports and "Q" not in ports and "Y" not in ports:
                if "QN" in ports:
                    outputs.append(ports["QN"])
            elif "Y" in ports:
                outputs.append(ports["Y"])
            elif "ZN" in ports:
                outputs.append(ports["ZN"])
            elif "Z" in ports:
                outputs.append(ports["Z"])
            # Derive the module type by inspecting the instance name.
            module_name = top_module_name
            inst_lower = instance_name.lower()
            if "multiplier" in inst_lower or "mul" in inst_lower:
                module_name = "multiplier"
            elif "adder" in inst_lower or "add" in inst_lower:
                module_name = "adder"
            elif "subtractor" in inst_lower or "sub" in inst_lower:
                module_name = "subtractor"
            elif "comparator" in inst_lower:
                module_name = "comparator"
            # Append the gate to the dataset split list
            if trial == 3:
                local_tr.append(ml_count)
            elif trial == 1:
                local_va.append(ml_count)
            else:
                local_te.append(ml_count)
            # Build a new node
            node = CircuitNode(
                name=instance_name,
                bool_func=bool_fun,
                inputs=inputs,
                outputs=outputs,
                processed=module_name,
                count=ml_count,
            )
            # If any of the gate outputs is a primary output, record
            # that it drives a PO
            for out_signal in outputs:
                if out_signal in netlist_outputs_set:
                    current_fwd = node.get_fwdgates()
                    current_fwd_inst = node.get_fwdgates_inst()
                    current_fwd.append("PO")
                    current_fwd_inst.append(out_signal)
                    node.set_fwdgates(current_fwd)
                    node.set_fwdgates_inst(current_fwd_inst)
            local_list_of_gates.append(instance_name)
            local_circuit[instance_name] = node
            ml_count += 1
            i += 1
            continue
        # If no pattern matched, move to the next line
        i += 1
    return (
        local_circuit,
        local_list_of_gates,
        local_tr,
        local_va,
        local_te,
        netlist_inputs_set,
        ml_count,
        assign_count,
    )


def connect_gates(list_of_gates: List[str],
                  the_circuit: Dict[str, CircuitNode],
                  netlist_inputs_set: set,
                  substr: str = "KEYINPUT",
                  substr2: str = "KEY_INPUTS") -> None:
    """Establish forward/backward connections between circuit nodes.

    For each node in ``the_circuit`` we inspect its input signals and
    determine whether they originate from a primary input (PI), a key
    input (KI) or the output of another gate.  The node's
    ``fedbygates`` and ``fedbygates_inst`` fields are populated
    accordingly.  The corresponding ``fwdgates`` and
    ``fwdgates_inst`` fields on upstream nodes are also updated so
    that each gate knows which gates it drives.  Connections to
    primary outputs are represented by the string ``"PO"`` in the
    ``fwdgates`` list and the signal name in ``fwdgates_inst``.

    Parameters
    ----------
    list_of_gates : List[str]
        List of gate instance names in the order they were created.
    the_circuit : Dict[str, CircuitNode]
        Mapping from instance names to ``CircuitNode`` objects.
    netlist_inputs_set : set
        Set of primary input signal names for the current netlist.
    substr : str
        Substring used to detect key input signal names (default
        ``"KEYINPUT"``).
    substr2 : str
        Additional substring used to detect key input signal names
        (default ``"KEY_INPUTS"``).
    """
    # Iterate over each node to build its incoming connections
    for node_name, node in the_circuit.items():
        name = node.get_name()
        inputs = node.get_inputs()
        # These lists will collect the boolean functions and instance
        # names of nodes feeding into the current node
        fed_by_types: List[str] = []
        fed_by_inst: List[str] = []
        # Separate inputs into primary inputs / key inputs and others
        remaining_inputs: List[str] = []
        for sig in inputs:
            # Remove any leading/trailing whitespace
            signal = sig.strip()
            # Check for primary inputs
            if signal in netlist_inputs_set:
                if substr in signal or substr2 in signal:
                    # Key input
                    fed_by_types.append("KI")
                    fed_by_inst.append(signal)
                else:
                    # Plain primary input
                    fed_by_types.append("PI")
                    fed_by_inst.append(signal)
            else:
                remaining_inputs.append(signal)
        # For each remaining input, search the outputs of existing gates
        # to find the producer gate
        if remaining_inputs:
            for upstream_name in list_of_gates:
                upstream_node = the_circuit.get(upstream_name)
                if upstream_node is None:
                    continue
                upstream_outputs = upstream_node.get_outputs()
                upstream_type = upstream_node.get_bool_func()
                for out_signal in upstream_outputs:
                    for signal in list(remaining_inputs):
                        if signal == out_signal:
                            # Record the feed for the current node
                            fed_by_types.append(upstream_type)
                            fed_by_inst.append(upstream_name)
                            # Update forward lists on upstream node
                            current_fwd = upstream_node.get_fwdgates()
                            current_fwd_inst = upstream_node.get_fwdgates_inst()
                            # Append and filter out None values later
                            current_fwd.append(node.get_bool_func())
                            current_fwd_inst.append(node_name)
                            # Persist the updates on the upstream node
                            upstream_node.set_fwdgates(
                                [x for x in current_fwd if x is not None]
                            )
                            upstream_node.set_fwdgates_inst(
                                [x for x in current_fwd_inst if x is not None]
                            )
                            # Remove this input so we don't match it again
                            remaining_inputs.remove(signal)
                            break
        # Assign the collected feed information to the node
        node.set_fedbygates([x for x in fed_by_types if x is not None])
        node.set_fedbygates_inst([x for x in fed_by_inst if x is not None])


def compute_features_and_edges(the_circuit: Dict[str, CircuitNode],
                               list_of_gates: List[str],
                               features_map: Dict[str, int],
                               module_map: Dict[str, int],
                               tr_set: set,
                               row_fh,
                               col_fh,
                               row_tr_fh,
                               col_tr_fh,
                               feat_fh,
                               label_fh,
                               cell_fh,
                               count_fh,
                               input_file_name: str) -> None:
    """Compute feature vectors and write graph edges for all nodes.

    This function closely follows the logic of the Perl script when
    computing the 34‑dimensional feature vector for each node and
    emitting the adjacency lists.  The feature vector encodes the
    gate's boolean function, the types of neighbouring gates and
    degree information.  Edges are written in both directions to
    ``row.txt``/``col.txt`` and, where appropriate, to
    ``row_tr.txt``/``col_tr.txt``.

    Parameters
    ----------
    the_circuit : Dict[str, CircuitNode]
        Dictionary of all circuit nodes.
    list_of_gates : List[str]
        Ordered list of gate instance names.
    features_map : Dict[str, int]
        Mapping from boolean function names to feature indices.
    module_map : Dict[str, int]
        Mapping from module names to label indices.
    tr_set : set
        Set of node indices belonging to the training split.  Used to
        decide which edges to record in the training adjacency list.
    row_fh, col_fh : file handles
        Output files for the global adjacency list.
    row_tr_fh, col_tr_fh : file handles
        Output files for the training adjacency list.
    feat_fh : file handle
        Output file for node feature vectors.
    label_fh : file handle
        Output file for node labels.
    cell_fh : file handle
        Output file for node meta information (count, name, file).
    count_fh : file handle
        Output file for node indices.
    input_file_name : str
        Name of the netlist file from which these nodes originate.
    """
    for node_name in list_of_gates:
        node = the_circuit[node_name]
        # Initialize feature vector with zeros
        features = [0] * 34
        bool_fun = node.get_bool_func()
        # Bump feature corresponding to this gate's boolean function
        if bool_fun in features_map:
            idx = features_map[bool_fun]
            features[idx] += 1
        # Gather backward neighbours (gates feeding this node)
        fed_types = node.get_fedbygates()
        fed_inst = node.get_fedbygates_inst()
        # For each directly feeding gate, increment its type feature and
        # explore one level deeper to count the types of gates feeding
        # that gate (except PI/PO/KI)
        for inst_name in fed_inst:
            if inst_name in the_circuit:
                upstream_node = the_circuit[inst_name]
                upstream_type = upstream_node.get_bool_func()
                if upstream_type in features_map:
                    features[features_map[upstream_type]] += 1
                # One level further up
                for grand_type in upstream_node.get_fedbygates():
                    if grand_type in features_map and grand_type not in ("PI", "PO", "KI"):
                        features[features_map[grand_type]] += 1
        # Count presence of PI and KI in the immediate inputs
        input_params = set(fed_types)
        if "PI" in input_params:
            features[features_map["PI"]] += 1
        if "KI" in input_params:
            features[features_map["KEY"]] += 1
        # Set in_degree feature (number of feeding gates)
        in_degree = len(fed_inst)
        features[features_map["in_degree"]] = in_degree
        # Gather forward neighbours (gates fed by this node)
        fwd_inst = [inst for inst in node.get_fwdgates_inst() if inst is not None]
        # out_degree feature counts the number of forward gates
        out_degree = len(fwd_inst)
        features[features_map["out_degree"]] = out_degree
        # For each forward gate, increment the feature corresponding to
        # that gate's type as well as the types of gates it feeds (one
        # level deeper) except PI/PO/KI
        for inst_name in fwd_inst:
            if inst_name in the_circuit:
                downstream_node = the_circuit[inst_name]
                downstream_type = downstream_node.get_bool_func()
                if downstream_type in features_map:
                    features[features_map[downstream_type]] += 1
                for grand_type in downstream_node.get_fwdgates():
                    if grand_type in features_map and grand_type not in ("PI", "PO", "KI"):
                        features[features_map[grand_type]] += 1
        # If this node feeds a primary output directly, mark the PO feature
        params = set(node.get_fwdgates())
        if "PO" in params:
            features[features_map["PO"]] += 1
        # Determine the label for this node based on its module type
        module_name = node.get_processed()
        if module_name in module_map:
            label = module_map[module_name]
        else:
            # Fall back to adder (0) or other heuristics
            lower = module_name.lower()
            if "adder" in lower:
                label = 0
            elif "subtractor" in lower:
                label = 3
            elif "comparator" in lower:
                label = 4
            elif "multiplier" in lower:
                label = 1
            else:
                # default to 2 (misc or unknown type)
                label = 2
        # Write node metadata
        node_idx = node.get_count()
        cell_fh.write(f"{node_idx} {node.get_name()} from file {input_file_name}\n")
        count_fh.write(f"{node_idx}\n")
        label_fh.write(f"{label}\n")
        # Write features as space‑separated integers
        feat_fh.write(" ".join(str(x) for x in features) + "\n")
        # Write edges: for each forward connection, emit both
        # directions so the adjacency list is undirected
        for dst_name in fwd_inst:
            if dst_name in the_circuit:
                dst_node = the_circuit[dst_name]
                dst_idx = dst_node.get_count()
                # forward direction (u->v)
                row_fh.write(f"{node_idx}\n")
                col_fh.write(f"{dst_idx}\n")
                # reverse direction (v->u) to remove explicit direction
                row_fh.write(f"{dst_idx}\n")
                col_fh.write(f"{node_idx}\n")
                # If both nodes are in the training set, record the
                # edge in the training adjacency list
                if node_idx in tr_set and dst_idx in tr_set:
                    row_tr_fh.write(f"{node_idx}\n")
                    col_tr_fh.write(f"{dst_idx}\n")
                    row_tr_fh.write(f"{dst_idx}\n")
                    col_tr_fh.write(f"{node_idx}\n")
        # If this node directly feeds a primary output, add a self
        # connection (count->count) to the adjacency list.  This
        # mirrors the behaviour of the Perl script which writes an
        # undirected edge from a PO back to the node itself.
        if "PO" in params:
            row_fh.write(f"{node_idx}\n")
            col_fh.write(f"{node_idx}\n")
            if node_idx in tr_set:
                row_tr_fh.write(f"{node_idx}\n")
                col_tr_fh.write(f"{node_idx}\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Command‑line interface for the netlist to graph converter.

    Parses command‑line arguments, iterates over all netlist files in
    the provided directory, builds the circuit graph for each file and
    writes the aggregated outputs.  Any unknown flags cause the
    program to exit with an error.  Returns an exit status code.
    """
    parser = argparse.ArgumentParser(
        description="Convert gate‑level netlists into a graph dataset.",
        add_help=False,
    )
    parser.add_argument('-h', '-help', action='help', help='Show this help message and exit.')
    parser.add_argument('-v', '-version', action='store_true', help='Display version information and exit.')
    parser.add_argument('-i', '-input', dest='input_dir', help='Input directory containing Verilog files.', required=False)
    args, unknown = parser.parse_known_args(argv)
    if args.v:
        # Mimic the Perl script's version string
        print("netlist_to_graph_re.py 0.1 2021/11/23")
        return 0
    if unknown:
        sys.stderr.write(f"ERROR: Unknown option(s): {' '.join(unknown)}\n")
        return 1
    if not args.input_dir:
        sys.stderr.write("ERROR: Expect an input Verilog files!\n")
        parser.print_help(sys.stderr)
        return 1
    input_dir = args.input_dir
    # Initialise counters for unique node and assignment names
    ml_count = 0  # global node counter across files
    assign_count = 0  # counter for assign statements across files
    # Lists capturing node indices in each dataset split
    global_tr: List[int] = []
    global_va: List[int] = []
    global_te: List[int] = []
    # Feature and module mappings replicate those from the Perl script
    module_map: Dict[str, int] = {
        "adder": 0,
        "multiplier": 1,
        "subtractor": 3,
        "comparator": 4,
    }
    features_map: Dict[str, int] = {
        "PI": 0,
        "PO": 1,
        "KEY": 2,
        "XOR": 3,
        "XNOR": 4,
        "AND": 5,
        "OR": 6,
        "NAND": 7,
        "NOR": 8,
        "INV": 9,
        "BUF": 10,
        "BUFH": 10,
        "BUFZ": 10,
        "ADDF": 11,
        "AOI": 12,
        "OAI": 13,
        "MXIT": 14,
        "AO1B": 15,
        "AOI2XB": 16,
        "AO": 17,
        "OA": 18,
        "OAI2XB": 19,
        "in_degree": 20,
        "out_degree": 21,
        "TIELO": 22,
        "TIEHI": 23,
        "RF2R": 24,
        "RF1R": 25,
        "PREICG": 26,
        "POSTICG": 27,
        "M": 28,
        "A": 29,
        "FRICG": 30,
        "MXT": 31,
        "MX": 32,
        "ADDH": 33,
    }
    # Determine the list of input files from the directory
    try:
        # input_files = sorted([f for f in os.listdir(input_dir) if not f.startswith('.') and os.path.isfile(os.path.join(input_dir, f))])
        input_files = sorted([
        f for f in os.listdir(input_dir)
        if not f.startswith('.')
        and os.path.isfile(os.path.join(input_dir, f))
        and f.lower().endswith('.v')   # <--- only Verilog!
    ])
    except OSError as e:
        sys.stderr.write(f"FATAL ERROR: Cannot open {input_dir}: {e}\n")
        return 1
    start_time = time.time()
    # Open all output files once.  They will be closed automatically
    # when the context manager exits at the end of main().
    with open('row.txt', 'w') as fh_row, \
        open('col.txt', 'w') as fh_col, \
        open('row_tr.txt', 'w') as fh_row_tr, \
        open('col_tr.txt', 'w') as fh_col_tr, \
        open('feat.txt', 'w') as fh_feat, \
        open('label.txt', 'w') as fh_label, \
        open('cell.txt', 'w') as fh_cell, \
        open('count.txt', 'w') as fh_count, \
        open('va.txt', 'w') as fh_va, \
        open('te.txt', 'w') as fh_te, \
        open('tr.txt', 'w') as fh_tr:
        # Process each file individually
        for fname in input_files:
            # Determine dataset split based on filename prefix
            trial = 3  # default to training
            if fname.startswith('Valid'):
                trial = 1
            elif fname.startswith('Test'):
                trial = 2
            file_path = os.path.join(input_dir, fname)
            print(f"Reading the file {fname}")
            # Parse the file and build its own circuit representation
            (local_circuit,
             local_list_of_gates,
             local_tr,
             local_va,
             local_te,
             netlist_inputs_set,
             ml_count,
             assign_count) = parse_netlist_file(
                file_path=file_path,
                ml_count_start=ml_count,
                assign_count_start=assign_count,
                trial=trial,
                top_module_name=fname,
            )
            # Append local indices to global lists
            global_tr.extend(local_tr)
            global_va.extend(local_va)
            global_te.extend(local_te)
            # Establish connections within this circuit
            connect_gates(local_list_of_gates, local_circuit, netlist_inputs_set,
                          substr="KEYINPUT", substr2="KEY_INPUTS")
            # Compute features and edges for this circuit
            local_tr_set = set(local_tr)
            compute_features_and_edges(
                the_circuit=local_circuit,
                list_of_gates=local_list_of_gates,
                features_map=features_map,
                module_map=module_map,
                tr_set=local_tr_set,
                row_fh=fh_row,
                col_fh=fh_col,
                row_tr_fh=fh_row_tr,
                col_tr_fh=fh_col_tr,
                feat_fh=fh_feat,
                label_fh=fh_label,
                cell_fh=fh_cell,
                count_fh=fh_count,
                input_file_name=fname,
            )
        # After processing all files write the split indices
        for idx in global_va:
            fh_va.write(f"{idx}\n")
        for idx in global_te:
            fh_te.write(f"{idx}\n")
        for idx in global_tr:
            fh_tr.write(f"{idx}\n")
        # Print execution time to stderr
        run_time = time.time() - start_time
        sys.stderr.write(f"\nProgram completed in {int(run_time)} sec without error.\n\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())