graph [
  directed 1
  node [
    id 0
    label "INPUT_a[0]"
  ]
  node [
    id 1
    label "INPUT_a[1]"
  ]
  node [
    id 2
    label "INPUT_b[0]"
  ]
  node [
    id 3
    label "INPUT_b[1]"
  ]
  node [
    id 4
    label "OUTPUT_Result_mul[0]"
  ]
  node [
    id 5
    label "OUTPUT_Result_mul[1]"
  ]
  node [
    id 6
    label "OUTPUT_Result_mul[2]"
  ]
  node [
    id 7
    label "OUTPUT_Result_mul[3]"
  ]
  node [
    id 8
    label "OUTPUT_Result_add[0]"
  ]
  node [
    id 9
    label "OUTPUT_Result_add[1]"
  ]
  node [
    id 10
    label "\adder_1/U3"
    nodeType "OA21_X0P5M_A9TH"
  ]
  node [
    id 11
    label "\adder_1/U2"
    nodeType "XNOR3_X0P5M_A9TH"
  ]
  node [
    id 12
    label "\adder_1/U1"
    nodeType "NAND2_X1M_A9TH"
  ]
  node [
    id 13
    label "\multiplier_1/U6"
    nodeType "NOR2_X0P5M_A9TH"
  ]
  node [
    id 14
    label "\multiplier_1/U5"
    nodeType "AOI22_X1M_A9TH"
  ]
  node [
    id 15
    label "\multiplier_1/U4"
    nodeType "AND3_X1M_A9TH"
  ]
  node [
    id 16
    label "\multiplier_1/U3"
    nodeType "NOR2_X0P5M_A9TH"
  ]
  node [
    id 17
    label "\multiplier_1/U2"
    nodeType "NAND2_X1M_A9TH"
  ]
  node [
    id 18
    label "\multiplier_1/U1"
    nodeType "AND2_X1M_A9TH"
  ]
  edge [
    source 0
    target 11
  ]
  edge [
    source 0
    target 14
  ]
  edge [
    source 0
    target 15
  ]
  edge [
    source 0
    target 17
  ]
  edge [
    source 1
    target 10
  ]
  edge [
    source 1
    target 12
  ]
  edge [
    source 1
    target 14
  ]
  edge [
    source 1
    target 18
  ]
  edge [
    source 2
    target 11
  ]
  edge [
    source 2
    target 14
  ]
  edge [
    source 2
    target 15
  ]
  edge [
    source 2
    target 17
  ]
  edge [
    source 3
    target 10
  ]
  edge [
    source 3
    target 12
  ]
  edge [
    source 3
    target 14
  ]
  edge [
    source 3
    target 18
  ]
  edge [
    source 10
    target 9
  ]
  edge [
    source 11
    target 8
  ]
  edge [
    source 12
    target 10
  ]
  edge [
    source 12
    target 11
  ]
  edge [
    source 13
    target 6
  ]
  edge [
    source 14
    target 13
  ]
  edge [
    source 15
    target 4
  ]
  edge [
    source 15
    target 13
  ]
  edge [
    source 16
    target 5
  ]
  edge [
    source 17
    target 16
  ]
  edge [
    source 18
    target 7
  ]
  edge [
    source 18
    target 15
  ]
  edge [
    source 18
    target 16
  ]
]
