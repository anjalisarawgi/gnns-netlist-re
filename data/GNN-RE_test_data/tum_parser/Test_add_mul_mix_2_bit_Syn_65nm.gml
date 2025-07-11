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
    label "INPUT_c[0]"
  ]
  node [
    id 5
    label "INPUT_c[1]"
  ]
  node [
    id 6
    label "INPUT_d[0]"
  ]
  node [
    id 7
    label "INPUT_d[1]"
  ]
  node [
    id 8
    label "OUTPUT_Result[0]"
  ]
  node [
    id 9
    label "OUTPUT_Result[1]"
  ]
  node [
    id 10
    label "OUTPUT_Result[2]"
  ]
  node [
    id 11
    label "OUTPUT_Result[3]"
  ]
  node [
    id 12
    label "\adder_1/U3"
    nodeType "OA21_X1M_A9TH"
  ]
  node [
    id 13
    label "\adder_1/U2"
    nodeType "XNOR3_X0P5M_A9TH"
  ]
  node [
    id 14
    label "\adder_1/U1"
    nodeType "NAND2_X1M_A9TH"
  ]
  node [
    id 15
    label "\adder_2/U3"
    nodeType "OA21_X1M_A9TH"
  ]
  node [
    id 16
    label "\adder_2/U2"
    nodeType "XNOR3_X0P5M_A9TH"
  ]
  node [
    id 17
    label "\adder_2/U1"
    nodeType "NAND2_X1M_A9TH"
  ]
  node [
    id 18
    label "\multiplier_1/U6"
    nodeType "NOR2_X0P5M_A9TH"
  ]
  node [
    id 19
    label "\multiplier_1/U5"
    nodeType "NOR2_X0P5M_A9TH"
  ]
  node [
    id 20
    label "\multiplier_1/U4"
    nodeType "AND2_X1M_A9TH"
  ]
  node [
    id 21
    label "\multiplier_1/U3"
    nodeType "AOI22_X0P7M_A9TH"
  ]
  node [
    id 22
    label "\multiplier_1/U2"
    nodeType "NAND2_X0P7A_A9TH"
  ]
  node [
    id 23
    label "\multiplier_1/U1"
    nodeType "AND3_X0P7M_A9TH"
  ]
  edge [
    source 0
    target 13
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
    source 2
    target 13
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
    source 4
    target 16
  ]
  edge [
    source 5
    target 15
  ]
  edge [
    source 5
    target 17
  ]
  edge [
    source 6
    target 16
  ]
  edge [
    source 7
    target 15
  ]
  edge [
    source 7
    target 17
  ]
  edge [
    source 12
    target 20
  ]
  edge [
    source 12
    target 21
  ]
  edge [
    source 13
    target 21
  ]
  edge [
    source 13
    target 22
  ]
  edge [
    source 13
    target 23
  ]
  edge [
    source 14
    target 12
  ]
  edge [
    source 14
    target 13
  ]
  edge [
    source 15
    target 20
  ]
  edge [
    source 15
    target 21
  ]
  edge [
    source 16
    target 21
  ]
  edge [
    source 16
    target 22
  ]
  edge [
    source 16
    target 23
  ]
  edge [
    source 17
    target 15
  ]
  edge [
    source 17
    target 16
  ]
  edge [
    source 18
    target 10
  ]
  edge [
    source 19
    target 9
  ]
  edge [
    source 20
    target 11
  ]
  edge [
    source 20
    target 19
  ]
  edge [
    source 20
    target 23
  ]
  edge [
    source 21
    target 18
  ]
  edge [
    source 22
    target 19
  ]
  edge [
    source 23
    target 8
  ]
  edge [
    source 23
    target 18
  ]
]
