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
    label "AND2_X1_1"
  ]
  node [
    id 6
    label "AND2_X1_2"
  ]
  node [
    id 7
    label "BUF_X32_1"
  ]
  node [
    id 8
    label "BUF_X32_2"
  ]
  node [
    id 9
    label "OR2_X1_1"
  ]
  node [
    id 10
    label "XOR2_X1_1"
  ]
  node [
    id 11
    label "XOR2_X1_2"
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 0
    target 10
  ]
  edge [
    source 1
    target 5
  ]
  edge [
    source 1
    target 10
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
    target 9
  ]
  edge [
    source 6
    target 9
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
    target 7
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
    target 8
  ]
]
