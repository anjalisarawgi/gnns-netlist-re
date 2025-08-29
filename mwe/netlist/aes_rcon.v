/////////////////////////////////////////////////////////////////////
 
////                                                             ////
////  AES RCON Block                                             ////
////                                                             ////
////                                                             ////
////  Author: Rudolf Usselmann                                   ////
////          rudi@asics.ws                                      ////
////                                                             ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/cores/aes_core/  ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2000-2002 Rudolf Usselmann                    ////
////                         www.asics.ws                        ////
////                         rudi@asics.ws                       ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////


module aes_rcon(clk, kld, \out[0] , \out[1] , \out[2] , \out[3] , \out[4] , \out[5] , \out[6] , \out[7] , \out2[0] , \out2[1] , \out2[2] , \out2[3] , \out2[4] , \out2[5] , \out2[6] , \out2[7] );

input clk;
input kld;
output \out[0] ;
output \out[1] ;
output \out[2] ;
output \out[3] ;
output \out[4] ;
output \out[5] ;
output \out[6] ;
output \out[7] ;
output \out2[0] ;
output \out2[1] ;
output \out2[2] ;
output \out2[3] ;
output \out2[4] ;
output \out2[5] ;
output \out2[6] ;
output \out2[7] ;
INVX1 INVX1_1 ( .A(rcnt_reg_0_), .Y(_abc_1022_new_n26_));
NOR2X1 NOR2X1_1 ( .A(kld), .B(rcnt_reg_1_), .Y(rcnt_next_1_));
NAND2X1 NAND2X1_1 ( .A(_abc_1022_new_n26_), .B(rcnt_next_1_), .Y(_abc_1022_new_n28_));
INVX1 INVX1_2 ( .A(kld), .Y(_abc_1022_new_n29_));
INVX1 INVX1_3 ( .A(rcnt_reg_3_), .Y(_abc_1022_new_n30_));
NAND2X1 NAND2X1_2 ( .A(rcnt_reg_1_), .B(rcnt_reg_2_), .Y(_abc_1022_new_n31_));
NAND2X1 NAND2X1_3 ( .A(_abc_1022_new_n30_), .B(_abc_1022_new_n31_), .Y(_abc_1022_new_n32_));
NAND3X1 NAND3X1_1 ( .A(rcnt_reg_1_), .B(rcnt_reg_2_), .C(rcnt_reg_3_), .Y(_abc_1022_new_n33_));
NAND3X1 NAND3X1_2 ( .A(_abc_1022_new_n29_), .B(_abc_1022_new_n33_), .C(_abc_1022_new_n32_), .Y(_abc_1022_new_n34_));
AOI21X1 AOI21X1_1 ( .A(_abc_1022_new_n26_), .B(rcnt_reg_1_), .C(kld), .Y(_abc_1022_new_n35_));
OAI21X1 OAI21X1_1 ( .A(rcnt_reg_1_), .B(rcnt_reg_2_), .C(_abc_1022_new_n29_), .Y(_abc_1022_new_n36_));
INVX1 INVX1_4 ( .A(_abc_1022_new_n36_), .Y(_abc_1022_new_n37_));
NAND3X1 NAND3X1_3 ( .A(_abc_1022_new_n31_), .B(_abc_1022_new_n35_), .C(_abc_1022_new_n37_), .Y(_abc_1022_new_n38_));
INVX1 INVX1_5 ( .A(rcnt_reg_1_), .Y(_abc_1022_new_n39_));
OAI21X1 OAI21X1_2 ( .A(rcnt_reg_0_), .B(_abc_1022_new_n39_), .C(_abc_1022_new_n29_), .Y(_abc_1022_new_n40_));
INVX1 INVX1_6 ( .A(_abc_1022_new_n31_), .Y(_abc_1022_new_n41_));
OAI21X1 OAI21X1_3 ( .A(_abc_1022_new_n41_), .B(_abc_1022_new_n36_), .C(_abc_1022_new_n40_), .Y(_abc_1022_new_n42_));
NAND2X1 NAND2X1_4 ( .A(_abc_1022_new_n42_), .B(_abc_1022_new_n38_), .Y(_abc_1022_new_n43_));
NOR2X1 NOR2X1_2 ( .A(kld), .B(_abc_1022_new_n26_), .Y(rcnt_next_0_));
NAND2X1 NAND2X1_5 ( .A(_abc_1022_new_n41_), .B(rcnt_next_0_), .Y(_abc_1022_new_n45_));
OAI22X1 OAI22X1_1 ( .A(_abc_1022_new_n34_), .B(_abc_1022_new_n45_), .C(_abc_1022_new_n28_), .D(_abc_1022_new_n43_), .Y(\out[1] ));
NAND2X1 NAND2X1_6 ( .A(rcnt_reg_3_), .B(_abc_1022_new_n31_), .Y(_abc_1022_new_n47_));
NAND3X1 NAND3X1_4 ( .A(rcnt_reg_1_), .B(rcnt_reg_2_), .C(_abc_1022_new_n30_), .Y(_abc_1022_new_n48_));
AOI21X1 AOI21X1_2 ( .A(_abc_1022_new_n47_), .B(_abc_1022_new_n48_), .C(kld), .Y(rcnt_next_3_));
NAND3X1 NAND3X1_5 ( .A(_abc_1022_new_n42_), .B(rcnt_next_3_), .C(_abc_1022_new_n38_), .Y(_abc_1022_new_n50_));
NAND3X1 NAND3X1_6 ( .A(_abc_1022_new_n42_), .B(_abc_1022_new_n34_), .C(_abc_1022_new_n38_), .Y(_abc_1022_new_n51_));
NAND2X1 NAND2X1_7 ( .A(rcnt_reg_0_), .B(rcnt_next_1_), .Y(_abc_1022_new_n52_));
OAI22X1 OAI22X1_2 ( .A(_abc_1022_new_n52_), .B(_abc_1022_new_n51_), .C(_abc_1022_new_n28_), .D(_abc_1022_new_n50_), .Y(\out[2] ));
NOR2X1 NOR2X1_3 ( .A(_abc_1022_new_n41_), .B(_abc_1022_new_n36_), .Y(rcnt_next_2_));
NAND3X1 NAND3X1_7 ( .A(_abc_1022_new_n40_), .B(_abc_1022_new_n34_), .C(rcnt_next_2_), .Y(_abc_1022_new_n55_));
OAI21X1 OAI21X1_4 ( .A(_abc_1022_new_n34_), .B(_abc_1022_new_n45_), .C(_abc_1022_new_n55_), .Y(\out[3] ));
OR2X2 OR2X2_1 ( .A(rcnt_reg_1_), .B(rcnt_reg_2_), .Y(_abc_1022_new_n57_));
NAND3X1 NAND3X1_8 ( .A(_abc_1022_new_n29_), .B(_abc_1022_new_n31_), .C(_abc_1022_new_n57_), .Y(_abc_1022_new_n58_));
AOI21X1 AOI21X1_3 ( .A(_abc_1022_new_n40_), .B(_abc_1022_new_n58_), .C(_abc_1022_new_n34_), .Y(_abc_1022_new_n59_));
INVX1 INVX1_7 ( .A(rcnt_next_0_), .Y(_abc_1022_new_n60_));
OAI21X1 OAI21X1_5 ( .A(_abc_1022_new_n39_), .B(_abc_1022_new_n60_), .C(_abc_1022_new_n28_), .Y(_abc_1022_new_n61_));
NAND3X1 NAND3X1_9 ( .A(_abc_1022_new_n38_), .B(_abc_1022_new_n61_), .C(_abc_1022_new_n59_), .Y(_abc_1022_new_n62_));
NAND2X1 NAND2X1_8 ( .A(_abc_1022_new_n42_), .B(rcnt_next_3_), .Y(_abc_1022_new_n63_));
OAI21X1 OAI21X1_6 ( .A(_abc_1022_new_n40_), .B(_abc_1022_new_n58_), .C(_abc_1022_new_n34_), .Y(_abc_1022_new_n64_));
NOR2X1 NOR2X1_4 ( .A(_abc_1022_new_n39_), .B(_abc_1022_new_n60_), .Y(_abc_1022_new_n65_));
NAND3X1 NAND3X1_10 ( .A(_abc_1022_new_n65_), .B(_abc_1022_new_n64_), .C(_abc_1022_new_n63_), .Y(_abc_1022_new_n66_));
NAND2X1 NAND2X1_9 ( .A(_abc_1022_new_n62_), .B(_abc_1022_new_n66_), .Y(\out[4] ));
INVX1 INVX1_8 ( .A(_abc_1022_new_n28_), .Y(_abc_1022_new_n68_));
NAND3X1 NAND3X1_11 ( .A(_abc_1022_new_n68_), .B(_abc_1022_new_n38_), .C(_abc_1022_new_n59_), .Y(_abc_1022_new_n69_));
NAND3X1 NAND3X1_12 ( .A(_abc_1022_new_n68_), .B(_abc_1022_new_n64_), .C(_abc_1022_new_n63_), .Y(_abc_1022_new_n70_));
NAND2X1 NAND2X1_10 ( .A(_abc_1022_new_n69_), .B(_abc_1022_new_n70_), .Y(\out[5] ));
AOI21X1 AOI21X1_4 ( .A(_abc_1022_new_n35_), .B(rcnt_next_2_), .C(rcnt_next_3_), .Y(_abc_1022_new_n72_));
NOR3X1 NOR3X1_1 ( .A(_abc_1022_new_n52_), .B(_abc_1022_new_n59_), .C(_abc_1022_new_n72_), .Y(\out[6] ));
NOR2X1 NOR2X1_5 ( .A(_abc_1022_new_n34_), .B(_abc_1022_new_n42_), .Y(\out[7] ));
INVX1 INVX1_9 ( .A(_abc_1022_new_n42_), .Y(\out2[0] ));
OAI21X1 OAI21X1_7 ( .A(_abc_1022_new_n34_), .B(_abc_1022_new_n42_), .C(_abc_1022_new_n45_), .Y(\out2[1] ));
OR2X2 OR2X2_2 ( .A(_abc_1022_new_n34_), .B(_abc_1022_new_n45_), .Y(_abc_1022_new_n77_));
OAI21X1 OAI21X1_8 ( .A(_abc_1022_new_n28_), .B(_abc_1022_new_n51_), .C(_abc_1022_new_n77_), .Y(\out2[2] ));
INVX1 INVX1_10 ( .A(\out[7] ), .Y(_abc_1022_new_n79_));
OAI21X1 OAI21X1_9 ( .A(_abc_1022_new_n52_), .B(_abc_1022_new_n51_), .C(_abc_1022_new_n79_), .Y(\out2[3] ));
NAND3X1 NAND3X1_13 ( .A(_abc_1022_new_n77_), .B(_abc_1022_new_n55_), .C(_abc_1022_new_n79_), .Y(\out2[4] ));
NAND2X1 NAND2X1_11 ( .A(_abc_1022_new_n77_), .B(_abc_1022_new_n66_), .Y(\out2[5] ));
INVX1 INVX1_11 ( .A(_abc_1022_new_n70_), .Y(\out2[6] ));
OAI21X1 OAI21X1_10 ( .A(_abc_1022_new_n26_), .B(_abc_1022_new_n31_), .C(_abc_1022_new_n29_), .Y(\out[0] ));
NOR3X1 NOR3X1_2 ( .A(_abc_1022_new_n52_), .B(_abc_1022_new_n59_), .C(_abc_1022_new_n72_), .Y(\out2[7] ));
DFFPOSX1 DFFPOSX1_1 ( .CLK(clk), .D(rcnt_next_0_), .Q(rcnt_reg_0_));
DFFPOSX1 DFFPOSX1_2 ( .CLK(clk), .D(rcnt_next_1_), .Q(rcnt_reg_1_));
DFFPOSX1 DFFPOSX1_3 ( .CLK(clk), .D(rcnt_next_2_), .Q(rcnt_reg_2_));
DFFPOSX1 DFFPOSX1_4 ( .CLK(clk), .D(rcnt_next_3_), .Q(rcnt_reg_3_));


endmodule