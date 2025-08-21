graph [
  directed 1
  node [
    id 0
    label "INPUT_pci_frame_in"
  ]
  node [
    id 1
    label "INPUT_pci_irdy_in"
  ]
  node [
    id 2
    label "OUTPUT_pci_stop_out"
  ]
  node [
    id 3
    label "INPUT_stop_w"
  ]
  node [
    id 4
    label "INPUT_stop_w_frm"
  ]
  node [
    id 5
    label "INPUT_stop_w_frm_irdy"
  ]
  node [
    id 6
    label "AOI21X1_1"
  ]
  node [
    id 7
    label "INVX1_1"
  ]
  node [
    id 8
    label "INVX1_2"
  ]
  node [
    id 9
    label "NOR2X1_1"
  ]
  node [
    id 10
    label "OAI21X1_1"
  ]
  edge [
    source 0
    target 9
  ]
  edge [
    source 0
    target 10
  ]
  edge [
    source 1
    target 9
  ]
  edge [
    source 3
    target 8
  ]
  edge [
    source 4
    target 7
  ]
  edge [
    source 5
    target 6
  ]
  edge [
    source 6
    target 2
  ]
  edge [
    source 7
    target 10
  ]
  edge [
    source 8
    target 10
  ]
  edge [
    source 9
    target 6
  ]
  edge [
    source 10
    target 6
  ]
]
