graph [
  directed 1
  node [
    id 0
    label "INPUT_a"
  ]
  node [
    id 1
    label "INPUT_b"
  ]
  node [
    id 2
    label "INPUT_operation"
  ]
  node [
    id 3
    label "OUTPUT_Result"
  ]
  node [
    id 4
    label "U2"
    nodeType "MX2_X0P5M_A9TH"
  ]
  node [
    id 5
    label "\adder_1/U1"
    nodeType "XOR2_X0P5M_A9TH"
  ]
  node [
    id 6
    label "\multiplier_1/U1"
    nodeType "AND2_X1M_A9TH"
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 0
    target 6
  ]
  edge [
    source 1
    target 5
  ]
  edge [
    source 1
    target 6
  ]
  edge [
    source 2
    target 4
  ]
  edge [
    source 4
    target 3
  ]
  edge [
    source 5
    target 4
  ]
  edge [
    source 6
    target 4
  ]
]
