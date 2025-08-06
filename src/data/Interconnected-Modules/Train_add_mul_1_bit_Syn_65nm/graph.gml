graph [
  directed 1
  node [
    id 0
    label "0"
    name "1 \adder_1/U1 from file Train_add_mul_1_bit_Syn_65nm.v"
    class_label 0
    features "[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,1,0]"
  ]
  node [
    id 1
    label "1"
    name "2 \multiplier_1/U1 from file Train_add_mul_1_bit_Syn_65nm.v"
    class_label 1
    features "[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,1,0]"
  ]
  node [
    id 2
    label "2"
    name "0 U2 from file Train_add_mul_1_bit_Syn_65nm.v"
    class_label 2
    features "[1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,0,0,0,0,0,0,0,0,0,0,1,0]"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 2
  ]
  edge [
    source 0
    target 0
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 2
    target 0
  ]
]
