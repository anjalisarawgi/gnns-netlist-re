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
    label "OUTPUT_Result[0]"
  ]
  node [
    id 5
    label "OUTPUT_Result[1]"
  ]
  node [
    id 6
    label "OUTPUT_Result[2]"
  ]
  node [
    id 7
    label "OUTPUT_Result[3]"
  ]
  node [
    id 8
    label "U5"
    nodeType "MX2_X0P5M_A9TH"
  ]
  node [
    id 9
    label "U6"
    nodeType "MX2_X0P5M_A9TH"
  ]
  node [
    id 10
    label "U7"
    nodeType "AND2_X0P5M_A9TH"
  ]
  node [
    id 11
    label "U8"
    nodeType "AND2_X0P5M_A9TH"
  ]
  node [
    id 12
    label "\adder_1/U4"
    nodeType "XNOR3_X0P5M_A9TH"
  ]
  node [
    id 13
    label "\adder_1/U3"
    nodeType "NAND2_X1M_A9TH"
  ]
  node [
    id 14
    label "\adder_1/U2"
    nodeType "OA21_X0P7M_A9TH"
  ]
  node [
    id 15
    label "\multiplier_1/U6"
    nodeType "AND3_X1M_A9TH"
  ]
  node [
    id 16
    label "\multiplier_1/U5"
    nodeType "AND2_X1M_A9TH"
  ]
  node [
    id 17
    label "\multiplier_1/U4"
    nodeType "NAND2_X0P7M_A9TH"
  ]
  node [
    id 18
    label "\multiplier_1/U3"
    nodeType "NOR2_X0P7A_A9TH"
  ]
  node [
    id 19
    label "\multiplier_1/U2"
    nodeType "NOR2_X0P7A_A9TH"
  ]
  node [
    id 20
    label "\multiplier_1/U1"
    nodeType "AOI22_X0P5M_A9TH"
  ]
  node [
    id 21
    label "\comparator_1/U3"
    nodeType "OAI22_X1M_A9TH"
  ]
  node [
    id 22
    label "\comparator_1/U2"
    nodeType "INV_X0P7M_A9TH"
  ]
  node [
    id 23
    label "\comparator_1/U1"
    nodeType "AO21B_X1M_A9TH"
  ]
  edge [
    source 0
    target 12
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
    source 0
    target 20
  ]
  edge [
    source 0
    target 22
  ]
  edge [
    source 1
    target 13
  ]
  edge [
    source 1
    target 14
  ]
  edge [
    source 1
    target 16
  ]
  edge [
    source 1
    target 20
  ]
  edge [
    source 1
    target 23
  ]
  edge [
    source 2
    target 12
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
    source 2
    target 20
  ]
  edge [
    source 2
    target 21
  ]
  edge [
    source 2
    target 23
  ]
  edge [
    source 3
    target 13
  ]
  edge [
    source 3
    target 14
  ]
  edge [
    source 3
    target 16
  ]
  edge [
    source 3
    target 20
  ]
  edge [
    source 3
    target 21
  ]
  edge [
    source 8
    target 6
  ]
  edge [
    source 9
    target 7
  ]
  edge [
    source 10
    target 4
  ]
  edge [
    source 11
    target 5
  ]
  edge [
    source 12
    target 8
  ]
  edge [
    source 13
    target 12
  ]
  edge [
    source 13
    target 14
  ]
  edge [
    source 14
    target 9
  ]
  edge [
    source 15
    target 10
  ]
  edge [
    source 15
    target 19
  ]
  edge [
    source 16
    target 9
  ]
  edge [
    source 16
    target 15
  ]
  edge [
    source 16
    target 18
  ]
  edge [
    source 17
    target 18
  ]
  edge [
    source 18
    target 11
  ]
  edge [
    source 19
    target 8
  ]
  edge [
    source 20
    target 19
  ]
  edge [
    source 21
    target 8
  ]
  edge [
    source 21
    target 9
  ]
  edge [
    source 21
    target 10
  ]
  edge [
    source 21
    target 11
  ]
  edge [
    source 22
    target 21
  ]
  edge [
    source 22
    target 23
  ]
  edge [
    source 23
    target 21
  ]
]
