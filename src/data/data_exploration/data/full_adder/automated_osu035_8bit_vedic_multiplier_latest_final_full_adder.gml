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
    label "INPUT_cin"
  ]
  node [
    id 3
    label "OUTPUT_cout"
  ]
  node [
    id 4
    label "OUTPUT_sum"
  ]
  node [
    id 5
    label "AND2X2_1"
  ]
  node [
    id 6
    label "AND2X2_2"
  ]
  node [
    id 7
    label "BUFX2_1"
  ]
  node [
    id 8
    label "BUFX2_2"
  ]
  node [
    id 9
    label "NOR2X1_1"
  ]
  node [
    id 10
    label "NOR2X1_2"
  ]
  node [
    id 11
    label "NOR2X1_3"
  ]
  node [
    id 12
    label "NOR2X1_4"
  ]
  node [
    id 13
    label "OR2X2_1"
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 0
    target 9
  ]
  edge [
    source 1
    target 5
  ]
  edge [
    source 1
    target 9
  ]
  edge [
    source 2
    target 6
  ]
  edge [
    source 2
    target 11
  ]
  edge [
    source 5
    target 10
  ]
  edge [
    source 5
    target 13
  ]
  edge [
    source 6
    target 12
  ]
  edge [
    source 6
    target 13
  ]
  edge [
    source 7
    target 3
  ]
  edge [
    source 8
    target 4
  ]
  edge [
    source 9
    target 10
  ]
  edge [
    source 10
    target 6
  ]
  edge [
    source 10
    target 11
  ]
  edge [
    source 11
    target 12
  ]
  edge [
    source 12
    target 8
  ]
  edge [
    source 13
    target 7
  ]
]
