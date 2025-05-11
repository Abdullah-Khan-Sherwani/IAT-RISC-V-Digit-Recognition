#define STDOUT 0xd0580000

.section .text
.global _start
## START YOUR CODE HERE

_start:

    la a0, input_matrix     # Address of image
    addi t0, x0, 784             # 28x28 = 784 pixels

    # ###############################TEST############################
    # la a0, input_matrix
    # li a1, 28
    # call printToLogVectorized
    # j _finish
    # ###############################TEST############################

main:
    # Prologue (RV32V - 32-bit)
    addi sp, sp, -32             # Allocate stack (32 bytes, 8-word aligned)
    sw   ra, 28(sp)              # Save return address (32-bit)
    sw   s0, 24(sp)              # Save preserved registers (32-bit)
    sw   s1, 20(sp)
    sw   s2, 16(sp)
    sw   s3, 12(sp)
    sw   s4, 8(sp)
    sw   s5, 4(sp)
    sw   s6, 0(sp)
    

   # Call the layer 
   # Addresses of the input matrx, the eight feature map output, the biases & the filters have been handled inside the subroutine
    la a0, input_matrix
    la a1, conv_filters
    la a2, filter_bias
    la a3, conv_output
    call Conv_layer 

    # ###############################TEST############################
    # la a0, conv_output
    # li a1, 24 # prints one filter output 
    # call printToLogVectorized
    # j _finish
    # ###############################TEST############################

    # Epilogue (RV32V - 32-bit)
    lw   s6, 0(sp)               # Restore s6 (32-bit)
    lw   s5, 4(sp)               # Restore s5 (32-bit)
    lw   s4, 8(sp)               # Restore s4 (32-bit)
    lw   s3, 12(sp)              # Restore s3 (32-bit)
    lw   s2, 16(sp)              # Restore s2 (32-bit)
    lw   s1, 20(sp)              # Restore s1 (32-bit)
    lw   s0, 24(sp)              # Restore s0 (32-bit)
    lw   ra, 28(sp)              # Restore return address (32-bit)
    addi sp, sp, 32              # Deallocate stack


# =============================================================================
# Linking Conv Layer with ReLU
# Done by SafeGOAT
# Scroll Down to see ReLU subroutine
# =============================================================================
# la a0, input_matrix # Testing


la a0, conv_output # Input to ReLU is the output of convolutional layer
li a1, 4608 # size of the output of the convolutional layer 24x24x8
addi sp, sp, -16    # Pre-align for ReLU's stack usage 
call relu_activation # Call ReLU subroutine
addi sp, sp, 16     # Rebalance stack  

# ##############################TEST############################
# la a0, conv_output
# li a1, 24
# call printToLogVectorized
# j _finish
# ###############################TEST############################


#la t0, test_matrix
la t0, conv_output       # base of input
la t1, output_max      # base of output
li t2, 24                 # input width
li t3, 12                 # output width
li t4, 8                  # depth count

call maxpool_2x2

# ##############################TEST############################
# la a0, output_max
# li a1, 34
# call printToLogVectorized
# j _finish
# ##############################TEST############################

call Flatten
# ###############################TEST############################
# la a0, output_max_flattened
# li a1, 34
# call printToLogVectorized
# j _finish
# ###############################TEST############################

dense_layer:

    # Prologue
    addi sp, sp, -20
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw ra, 16(sp)
    
    # Load base addresses
    la a0, dense_weights         # s2 = &W[0][0]
    la a1, output_max_flattened            # s3 = &A[0]
    la a2, dense_bias        # s4 = &b[0]
    la a3, dense_outputs         # s5 = &Z[0]

    # Initialize loop counters
    li s0, 0                   # s0 = i (output neuron index)
    li s3, 10                  # s3 = number of output neurons (10)

outer_loop:
    bge s0, s3, end_outer      # if i >= 10, exit outer loop
    
    # Initialize dot product accumulator
    mv t0, zero                # t0 = dot_product = 0.0
    fcvt.s.w ft0, t0           # ft0 = 0.0 (float)
    
    # Inner loop setup
    li s1, 0                   # s1 = j (input feature index)
    li s2, 1152                # s2 = number of input features (1152)
    
    # Calculate address of current row in W
    li t1, 1152                # t1 = 1152 (elements per row)
    mul t1, s0, t1             # t1 = i * 1152
    slli t1, t1, 2             # t1 = byte offset for row i
    add t1, a0, t1             # t1 = &W[i][0]
    
inner_loop:
    bge s1, s2, end_inner      # if j >= 1152, exit inner loop
    
    # Load W[i][j]
    lw t2, 0(t1)               # t2 = W[i][j] (as integer)
    fmv.w.x ft1, t2            # ft1 = W[i][j] (float)
    
    # Load A_flat[j]
    slli t3, s1, 2             # t3 = j * 4 (byte offset)
    add t3, a1, t3             # t3 = &A_flat[j]
    lw t4, 0(t3)               # t4 = A_flat[j] (as integer)
    fmv.w.x ft2, t4            # ft2 = A_flat[j] (float)
    
    # Multiply and accumulate
    fmadd.s ft0, ft1, ft2, ft0 # ft0 += W[i][j] * A_flat[j]
    
    # Increment pointers and counters
    addi t1, t1, 4             # move to next element in W row
    addi s1, s1, 1             # j++
    j inner_loop
    
end_inner:
    # Add bias term b[i]
    slli t5, s0, 2             # t5 = i * 4 (byte offset)
    add t5, a2, t5             # t5 = &b[i]
    lw t6, 0(t5)               # t6 = b[i] (as integer)
    fmv.w.x ft3, t6            # ft3 = b[i] (float)
    
    fadd.s ft0, ft0, ft3       # ft0 += b[i]
    
    # Store result in Z[i]
    slli t5, s0, 2             # t5 = i * 4 (byte offset)
    add t5, a3, t5             # t5 = &Z[i]
    fmv.x.w t6, ft0            # t6 = Z[i] (as integer)
    sw t6, 0(t5)               # store Z[i]
    
    # Next output neuron
    addi s0, s0, 1             # i++
    j outer_loop

end_outer:

    # Epilogue
    lw s7, 0(sp)
    lw s6, 4(sp)
    lw s5, 8(sp)
    lw s4, 12(sp)
    lw s3, 16(sp)
    lw s2, 20(sp)
    lw s1, 24(sp)
    lw s0, 28(sp)
    lw ra, 32(sp)
    addi sp, sp, 36

    la a0, dense_outputs
    li a1, 4
    call printToLogVectorized
    j _finish

    j softmax_layer

# SOFTMAX LAYER

# softmax_layer:
#     la a4, p
#     li t0, 0              # i = 0
#     flw f0, 0(a3)         # f0 = max = y[0]

# max_loop:
#     li t1, 10
#     bge t0, t1, compute_exp_sum

#     slli t2, t0, 2        # t2 = i * 4
#     add t3, a3, t2
#     flw f1, 0(t3)         # f1 = y[i]
#     fmax.s f0, f0, f1     # f0 = max(f0, f1)

#     addi t0, t0, 1
#     j max_loop


# Gpt given softmax below
.section .text
  .globl softmax_layer
softmax_layer:
    # --- Prologue: save caller-saves and temps ---
    addi  sp, sp, -40
    sw    ra,   36(sp)
    sw    s0,   32(sp)
    sw    s1,   28(sp)
    sw    s2,   24(sp)
    fsw   ft0,  20(sp)
    fsw   ft1,  16(sp)
    fsw   ft2,  12(sp)

    # --- Registers ---
    # a3 = logits_base
    # a4 = probs_base
    li    s2, 10          # vector length
    
    # --- Step 1: find max(logits) ---
    flw   ft0, 0(a3)      # ft0 = max = logits[0]
    li    s0, 1
1:  blt   s0, s2, 1b_endcheck
1b:
    slli  t0, s0, 2
    add   t1, a3, t0
    flw   ft1, 0(t1)      # ft1 = logits[s0]
    fmax.s ft0, ft0, ft1  # ft0 = max(ft0, ft1)
    addi  s0, s0, 1
    j     1b
1b_endcheck:

    # --- Step 2: compute exp_approx[i]=1+(x–max)+0.5*(x–max)^2, sum in ft2 ---
    fmv.s.x   ft2, zero   # ft2 = sum = 0.0
    li    s0, 0
2:  blt   s0, s2, 2b_endcheck
2b:
    slli  t0, s0, 2
    add   t1, a3, t0
    flw   ft1, 0(t1)      # ft1 = x

    # x_minus_max = ft1 − ft0
    fsub.s ft1, ft1, ft0  

    # exp ≈ 1 + x + 0.5 x^2
    li    t2, 0x3f800000  # 1.0f
    fmv.s.x ft2, t2       # reuse ft2 just as tmp to load 1.0
    fadd.s ft1, ft1, ft2  # ft1 = 1.0 + (x–max)

    fmul.s ft2, ft1, ft1  # ft2 = (x–max)^2
    li    t2, 0x3f000000  # 0.5f
    fmv.s.x ft3, t2
    fmul.s ft2, ft2, ft3  # ft2 = 0.5*(x–max)^2

    fadd.s ft1, ft1, ft2  # ft1 = 1 + (x–max) + 0.5(x–max)^2

    # store exp_approx[i] to probs_base
    add   t1, a4, t0
    fsw   ft1, 0(t1)

    # sum += exp_approx
    fadd.s ft2, ft2, ft1  # careful: reuse ft2; you may want another tmp if needed

    addi  s0, s0, 1
    j     2b
2b_endcheck:

    # --- Step 3: normalize each exp[i] by sum (in ft2) ---
    li    s0, 0
3:  blt   s0, s2, 3b_endcheck
3b:
    slli  t0, s0, 2
    add   t1, a4, t0
    flw   ft1, 0(t1)      # ft1 = exp_approx[i]
    fdiv.s ft1, ft1, ft2  # ft1 = prob[i] = exp_approx[i]/sum
    fsw   ft1, 0(t1)

    addi  s0, s0, 1
    j     3b
3b_endcheck:

    # --- Epilogue: restore all ---
    flw   ft2, 12(sp)
    flw   ft1, 16(sp)
    flw   ft0, 20(sp)
    lw    s2,   24(sp)
    lw    s1,   28(sp)
    lw    s0,   32(sp)
    lw    ra,   36(sp)
    addi  sp,   sp, 40
    ret

# =======================================
# Step 2: Compute exp(y[i] - max) and sum
# =======================================
compute_exp_sum:
    li t0, 0
    fmv.s.x f2, x0        # f2 = sum = 0.0

exp_loop:
    li t1, 10
    bge t0, t1, normalize_softmax

    slli t2, t0, 2
    add t3, a3, t2
    flw f1, 0(t3)         # f1 = y[i]
    fsub.s f3, f1, f0     # f3 = y[i] - max

    # Approximate exp(f3) ≈ 1 + x + 0.5 * x^2
    li t6, 0x3f800000     # float 1.0
    fmv.s.x f6, t6        # f6 = 1.0

    fadd.s f4, f6, f3     # f4 = 1.0 + x
    fmul.s f7, f3, f3     # f7 = x^2

    li t6, 0x3f000000     # float 0.5
    fmv.s.x f8, t6        # f8 = 0.5
    fmul.s f7, f7, f8     # f7 = 0.5 * x^2

    fadd.s f4, f4, f7     # f4 = 1 + x + 0.5 * x^2

    # Store exp_approx to p[i]
    add t3, a4, t2
    fsw f4, 0(t3)

    # Add to sum
    fadd.s f2, f2, f4     # sum += exp_approx

    addi t0, t0, 1
    j exp_loop

# =====================================
# Step 3: Normalize p[i] by dividing sum
# =====================================
normalize_softmax:
    li t0, 0

norm_loop:
    li t1, 10
    bge t0, t1, done1

    slli t2, t0, 2
    add t3, a4, t2
    flw f1, 0(t3)        # f1 = p[i] = exp_approx
    fdiv.s f1, f1, f2    # f1 = p[i] / sum
    fsw f1, 0(t3)        # store back p[i]

    addi t0, t0, 1
    j norm_loop

done1:
    flw fa0, 0(a4)
    flw fa1, 4(a4)
    flw fa2, 8(a4)
    flw fa3, 12(a4)
    flw fa4, 16(a4)
    flw fa5, 20(a4)
    flw fa6, 24(a4)
    flw fa7, 28(a4)
    flw fs2, 32(a4)
    flw fs3, 36(a4)
    
    move a0, a4
    #la a0, test
    li a1, 4
    call printToLogVectorized
    j _finish


#********************************************************************** Subroutines ************************************************************************************

#################################################################################
# Convolutional Layer Subroutine
# a0: input matrix
# a1: conv_filters address
# a2: conv filter biases address
# a3: conv_output address
#################################################################################
# # SCALAR
# Conv_layer:
# #Declarations  ------> Data from vectorized modified has been pulled here in the subroutine

#     li t0, 0          # filter index (0 to 7)
# filter_loop:
#     li t1, 0          # out_y = 0 to 23
#     li s0, 24         # reuse s0 as constant 24
#     # # At the end of filter_loop
#     # addi t0, t0, 1
#     # li s9, 8
#     # blt t0, s9, filter_loop
    
# out_y_loop:
#     li t2, 0          # out_x = 0 to 23
    
#     # # At the end of out_y_loop
#     # addi t1, t1, 1
#     # blt t1, s0, out_y_loop

    
# out_x_loop:
#     # calculate output offset = ((filter * 576) + (out_y * 24 + out_x)) * 4
#     mul t3, t0, s0        # t3 = filter * 24
#     mul t3, t3, s0        # t3 = filter * 576
#     mul t4, t1, s0        # t4 = out_y * 24
#     add t4, t4, t2        # t4 = out_y * 24 + out_x
#     add t5, t3, t4        # t5 = output index
#     slli t5, t5, 2        # t5 = offset in bytes
#     add t6, a3, t5        # t6 = output address

#     # Load bias for current filter
#     slli s10, t0, 2
#     add s10, a2, s10
#     flw ft3, 0(s10)        # ft3 = accumulator = bias

#     li s10, 0              # ky
    
#     # # At the end of out_x_loop
#     # addi t2, t2, 1
#     # blt t2, s0, out_x_loop     # s0 = 24
    
# conv_y_loop:
#     li t4, 0              # kx
# conv_x_loop:
#     # input_x = out_x + kx
#     add s11, t2, t4
#     # input_y = out_y + ky
#     add a4, t1, s10

#     # in_index = input_y * 28 + input_x
#     li s1, 28
#     mul s2, a4, s1
#     add s2, s2, s11
#     slli s2, s2, 2
#     add s3, a0, s2
#     flw ft0, 0(s3)        # ft0 = input pixel

#     # filter_offset = ((filter * 25) + (ky * 5 + kx)) * 4
#     li s4, 25
#     mul s5, t0, s4
#     li s6, 5
#     mul s7, s10, s6
#     add s7, s7, t4
#     add s5, s5, s7
#     slli s5, s5, 2
#     add s6, a1, s5
#     flw ft1, 0(s6)        # ft1 = filter weight

#     # Multiply and accumulate
#     fmul.s ft2, ft0, ft1
#     fadd.s ft3, ft3, ft2

#     addi t4, t4, 1
#     li s8, 5
#     blt t4, s8, conv_x_loop

#     addi s10, s10, 1
#     blt s10, s8, conv_y_loop

#     # Apply ReLU
#     # fmv.s.x ft0, zero
#     # fmax.s ft3, ft3, ft0

#     # Store result
#     fsw ft3, 0(t6)

#     # next out_x
#     addi t2, t2, 1
#     blt t2, s0, out_x_loop

#     # next out_y
#     addi t1, t1, 1
#     blt t1, s0, out_y_loop

#     # next filter
#     addi t0, t0, 1
#     li s9, 8
#     blt t0, s9, filter_loop

#     # exit
#     ret
# # exit:
# #    li a0, 10
# #    ecall

# VECTOR
Conv_layer:
    # Initialize filter index (0 to 7)
    li t0, 0                  # t0 = current filter index (0-7)
    
    filter_loop:
        # =============================================
        # Load bias for current filter
        # =============================================
        slli s10, t0, 2           # s10 = filter_index * 4 (float size)
        add s10, a2, s10          # s10 = address of bias[filter]
        flw ft3, 0(s10)           # ft3 = bias value for current filter
        
        # Initialize output row index (0 to 23)
        li t1, 0                  # t1 = out_y (vertical position in output)
        li s0, 28                 # s0 = constant 28 (input dimension)
        li s1, 24                 # s1 = constant 24 (output dimension)
        
        # Configure vector unit (4 elements per vector)
        # li t3, 4                  # Process 4 outputs simultaneously
        li t3, 8                  # Process 8 outputs simultaneously
        vsetvli t3, t3, e32, m1   # Set vector length to 4 or 8, 32-bit floats
        
        out_y_loop:
            # Initialize output column index (process 4 columns at a time)
            li t2, 0                  # t2 = out_x (horizontal position in output)
            
            out_x_loop:
                # Initialize vector accumulator with bias value
                vfmv.v.f v4, ft3          # v4 = [bias, bias, bias, bias] (vector)
                
                # Process all positions in the 5x5 filter for this output position
                li s10, 0                 # s10 = ky (filter row index)
                
                conv_y_loop:
                    li s11, 0                 # s11 = kx (filter column index)
                    
                    conv_x_loop:
                        # =============================================
                        # Load filter weight for current position
                        # Filter memory layout: [filter][ky][kx]
                        # =============================================
                        li s4, 25                 # 25 weights per filter (5x5)
                        mul s5, t0, s4            # s5 = filter_index * 25
                        li s6, 5                  # Filter width = 5
                        mul s7, s10, s6           # s7 = ky * 5
                        add s5, s5, s7            # s5 = filter*25 + ky*5
                        add s5, s5, s11           # s5 += kx (current filter column)
                        slli s5, s5, 2            # Convert to byte offset
                        add s6, a1, s5            # s6 = &filter[filter][ky][kx]
                        flw ft1, 0(s6)            # ft1 = weight at (ky, kx) (scalar)
                        
                        # =============================================
                        # Load 4 or 8 input pixels horizontally for current filter position
                        # Input memory layout: [input_y][input_x]
                        # =============================================
                        add a4, t1, s10           # a4 = input_y = out_y + ky
                        add a5, t2, s11           # a5 = base input_x = out_x + kx
                        li s2, 28                 # Input width = 28
                        mul s3, a4, s2            # s3 = input_y * 28
                        add s3, s3, a5            # s3 += base input_x
                        slli s3, s3, 2            # Convert to byte offset
                        add s7, a0, s3            # s7 = &input[input_y][input_x]
                        vle32.v v5, (s7)          # v5 = [input[y][x], input[y][x+1], input[y][x+2], input[y][x+3]] more elements if using 8 wide vectors
                        
                        # =============================================
                        # Vector multiply-accumulate operation
                        # =============================================
                        vfmv.v.f v6, ft1          # Broadcast filter weight to all vector lanes
                        vfmacc.vv v4, v5, v6      # v4 += v5 * v6 (element-wise)
                        
                        # Next filter column
                        addi s11, s11, 1          # kx++
                        li s8, 5
                        blt s11, s8, conv_x_loop  # Loop until kx = 5
                    
                    # Next filter row
                    addi s10, s10, 1          # ky++
                    li s8, 5
                    blt s10, s8, conv_y_loop  # Loop until ky = 5
                
                # =============================================
                # Calculate output address and store results
                # Output memory layout: [filter][out_y][out_x]
                # =============================================
                li s4, 576                # 24*24 outputs per filter
                mul s5, t0, s4            # s5 = filter_index * 576
                mul s6, t1, s1            # s6 = out_y * 24
                add s6, s6, t2            # s6 += out_x
                add s5, s5, s6            # s5 = absolute output index
                slli s5, s5, 2            # Convert to byte offset
                add s7, a3, s5            # s7 = &output[filter][out_y][out_x]
                
                # Store 4 or 8 results
                vse32.v v4, (s7)          # Store vector to output
                
                # Next block of 4 or 8 columns
                # addi t2, t2, 4            # out_x += 4
                addi t2, t2, 8            # out_x += 8

                # Make sure we don't exceed the output width
                # li s9, 20                 # 24-4 = 20 (last starting position for 4-wide vector)
                li s9, 16                 # 24-8 = 16 (last starting position for 8-wide vector)
                ble t2, s9, out_x_loop    # Loop while out_x <= 20
            
            # Check if we need to handle the last columns
            # bge t2, s1, skip_last_columns # Skip if we've processed all columns
            bge t2, s1, end_row # Skip if we've processed all columns
            
            # Handle remaining columns (t2=20 to t2=23)
            # For simplicity, we'll use the vector operations with masking
            vfmv.v.f v4, ft3          # Reset accumulator with bias
            
            # Process all positions in the 5x5 filter for remaining output position
            li s10, 0                 # Reset ky (filter row index)

            # Handle remaining columns (if needed, or you can skip this)
            # ...
            end_row:
                # Next output row
                addi t1, t1, 1            # out_y++
                blt t1, s1, out_y_loop    # Loop until out_y = 24
            
            # Next filter
            addi t0, t0, 1            # filter_index++
            li s9, 8
            blt t0, s9, filter_loop   # Loop until filter_index = 8

        end:
            ret                       # Return from function
################################# Conv2D Subroutine End ##################################################

################################# ReLU Subroutine ##################################################
# Both SCALAR and VECTOR built by SafeGOAT
# Prologue and epilogue of SCALAR are incomplete and atomic for VECTOR implementation
# Arguments:
#   a0: Matrix to be processed (conv_output in our case)
#   a1: Number of Elements that are to be processed
#   
# 
####################################################################################################
# SCALAR
# relu_activation:
#     # PROLOGUE
#     addi sp, sp, -16        # Allocate stack space
#     sw ra, 12(sp)           # Save return address
#     fsw fs0, 8(sp)          # Save preserved FP register
#     sw s0, 4(sp)            # Save preserved integer register
 
#     # Load 0.0 using stack space
#     addi sp, sp, -4         # Temporary space for 0.0
#     sw zero, 0(sp)
#     flw ft0, 0(sp)          # ft0 = 0.0
#     addi sp, sp, 4          # Deallocate temp space

#     # Initialize counter
#     li t0, 0                # t0 = counter
#     mv s0, a0               # s0 = preserved matrix pointer
#     fmv.s fs0, ft0          # fs0 = preserved 0.0

# relu_loop:
#     flw ft1, 0(s0)          # Load current element
#     fmax.s ft1, ft1, fs0    # ReLU: max(x, 0)
#     fsw ft1, 0(s0)          # Store result back
#     addi s0, s0, 4          # Next element
#     addi t0, t0, 1          # Increment counter
#     blt t0, a1, relu_loop   # Loop if not done



#     # EPILOGUE
#     lw s0, 4(sp)            # Restore saved register
#     flw fs0, 8(sp)          # Restore FP register
#     lw ra, 12(sp)           # Restore return address
#     addi sp, sp, 16         # Deallocate stack space
#     ret                     # Return to caller

# VECTOR
relu_activation:
    # PROLOGUE
    addi sp, sp, -32        # Allocate stack space for:
    sw ra, 28(sp)           # Return address
    sw s0, 24(sp)           # Preserved register s0
    fsw fs0, 20(sp)         # Preserved FP register fs0
    sw a0, 16(sp)           # Original pointer (a0)
    sw t1, 12(sp)           # Counter register t1
    sw t2, 8(sp)            # Temporary register t2
    sw a2, 4(sp)            # Vector length register a2

    # VECTOR IMPLEMENTATION OF RELU
    # Initialize zero register
    fcvt.s.w fs0, zero      # fs0 = 0.0 (more reliable than stack method)

    # Setup vector processing
    li t1, 18432            # t1 = total bytes (4608 floats * 4)
    mv s0, a0               # Save original pointer in preserved register

vector_loop:
    # Set vector length (elements processed per iteration)
    vsetvli a2, t1, e32     # a2 = elements this iteration (max possible)
                            # t1 = remaining bytes

    # Load vector of floats from memory
    vle32.v v8, (a0)        # Load VL floats into v8
    vfmax.vf v8, v8, fs0    # Vector-scalar max (broadcasts fs0), ReLU: v8 = max(v8, 0.0)

    # Store back to memory
    vse32.v v8, (a0)        # Store VL floats back full chilling

    # Update pointer & counter
    slli t2, a2, 2          # t2 = bytes processed (a2 * 4)
    add a0, a0, t2          # Move pointer forward
    sub t1, t1, t2          # Decrement remaining bytes
    bnez t1, vector_loop    # Loop if bytes remain

    # EPILOGUE
    lw a2, 4(sp)            # Restore a2
    lw t2, 8(sp)            # Restore t2
    lw t1, 12(sp)           # Restore t1
    lw a0, 16(sp)           # Restore original a0
    flw fs0, 20(sp)         # Restore fs0
    lw s0, 24(sp)           # Restore s0
    lw ra, 28(sp)           # Restore return address
    addi sp, sp, 32         # Deallocate stack space
    ret                     # Return to caller
################################# ReLU Subroutine end ##################################################


################################# MaxPool Subroutine ##################################################
#!!New Note!! -> Maxpool layer changed to accomodate 24*24 matrix from conv layer
# Make sure there is enough space allocated for the Maxpool layer
# The intern has loaded necessary addresses in the main function.
#######################################################################################################
# SCALAR
# maxpool_2x2:

# li a5, 2

# li s2, 0                  # depth/channel index
# loop_depth:
#     li s0, 0                  # output row
# loop_i:
#     li s1, 0                  # output col
# loop_j:
#     # input_index = s2*576 + (2*s0)*24 + 2*s1
#     mul a0, s0, a5            # a0 = 2*s0
#     mul a0, a0, t2            # a0 = 2*s0 * 24
#     slli a1, s1, 1            # a1 = 2*s1
#     add a0, a0, a1            # a0 = offset within 2D plane
#     li a2, 576                # 24*24
#     mul a3, s2, a2            # a3 = offset for depth slice
#     add a0, a0, a3            # a0 = total element offset
#     slli a0, a0, 2            # byte offset
#     add a1, t0, a0            # a1 = address of top-left of 2x2 patch

#     # Load 2x2 block
#     flw ft0, 0(a1)
#     flw ft1, 4(a1)
#     addi a2, a1, 96           # 24 * 4 bytes = next row
#     flw ft2, 0(a2)
#     flw ft3, 4(a2)

#     fmax.s ft0, ft0, ft1
#     fmax.s ft2, ft2, ft3
#     fmax.s ft0, ft0, ft2

#     # Store into output
#     mul a0, s0, t3            # a0 = s0 * 12
#     add a0, a0, s1            # a0 = output element index
#     li a2, 144                # 12*12
#     mul a3, s2, a2            # depth offset
#     add a0, a0, a3            # total output index
#     slli a0, a0, 2            # byte offset
#     add a1, t1, a0
#     fsw ft0, 0(a1)

#     addi s1, s1, 1
#     blt s1, t3, loop_j

#     addi s0, s0, 1
#     blt s0, t3, loop_i

#     addi s2, s2, 1
#     blt s2, t4, loop_depth

# end:
#     ret

# VECTOR:
maxpool_2x2:
    # Prologue
    addi sp, sp, -16          # Allocate space for 4 floats
    li a5, 2                  # Pool size = 2
    li s2, 0                  # Depth/channel index
    
    # We assume the following registers are set before calling this function:
    # t0 = input data pointer
    # t1 = output data pointer
    # t2 = input width (24)
    # t3 = output width (12)
    # t4 = depth/channels
    
    # Configure vector parameters
    li t5, 4                  # Process 4 elements
    vsetvli t6, t5, e32, m1   # Set vector length to 4, 32-bit elements
    
loop_depth_vector:
    li s0, 0                  # Output row
    
loop_i_vector:
    li s1, 0                  # Output col
    
loop_j_vector:
    # Calculate base address for current 2x2 block
    # Input position for top-left of 2x2 block:
    # (s0*2)*input_width + (s1*2) = s0*2*t2 + s1*2
    slli a0, s0, 1            # a0 = s0*2 (row stride)
    mul a0, a0, t2            # a0 = (s0*2)*input_width
    slli a1, s1, 1            # a1 = s1*2 (column offset)
    add a0, a0, a1            # a0 = offset within 2D plane
    
    # Calculate depth offset
    mul a2, t2, t2            # a2 = input plane size (input_width * input_width)
    mul a3, s2, a2            # a3 = depth offset
    add a0, a0, a3            # a0 = total element offset
    slli a0, a0, 2            # a0 = byte offset
    add a1, t0, a0            # a1 = base address
    
    # Load values from the 2x2 block
    # Top-left
    flw ft0, 0(a1)
    fsw ft0, 0(sp)
    
    # Top-right
    flw ft1, 4(a1)
    fsw ft1, 4(sp)
    
    # Bottom-left
    slli a4, t2, 2             # a4 = bytes per row (input_width * 4)
    add a2, a1, a4            # a2 = address of pixel below
    flw ft2, 0(a2)
    fsw ft2, 8(sp)
    
    # Bottom-right
    flw ft3, 4(a2)            # FIXED: Correct addressing for bottom-right element
    fsw ft3, 12(sp)
    
    # Load all 4 values into vector register
    vle32.v v0, (sp)          # v0 = [TL, TR, BL, BR]
    
    # Compute max of all elements
    vfredmax.vs v1, v0, v0    # Vector reduction max
    vfmv.f.s ft0, v1          # ft0 = max value
    
    # Calculate output address
    mul a0, s0, t3            # a0 = output_row * output_width
    add a0, a0, s1            # a0 = output element index
    mul a2, t3, t3            # output plane size
    mul a3, s2, a2            # depth offset
    add a0, a0, a3            # total output index
    slli a0, a0, 2            # byte offset
    add a1, t1, a0            # output address
    
    # Store result
    fsw ft0, 0(a1)
    
    addi s1, s1, 1
    blt s1, t3, loop_j_vector
    
    addi s0, s0, 1
    blt s0, t3, loop_i_vector
    
    addi s2, s2, 1
    blt s2, t4, loop_depth_vector
    
    # Epilogue
    addi sp, sp, 16
    ret
################################ MAXPOOL END ####################################

# FLATTENING MAXPOOL OUTPUT
Flatten:
    la a0, output_max_flattened
    li t2, 0           # Initialize column counter (original matrix)
    outer_loop1:
        li t3, 0           # Initialize row counter (original matrix)
    inner_loop1:
        # Calculate address in original matrix: t1 + (t3*8 + t2)*4
        # Since it's column-major: element at (row t3, col t2) is at t1 + (t3 + 144*t2)*4
        li t4, 144
        mul t4, t4, t2     # t4 = 144*t2
        add t4, t4, t3     # t4 = t3 + 144*t2
        slli t4, t4, 2     # t4 = (t3 + 144*t2)*4
        add t4, t1, t4     # t4 = t1 + (t3 + 144*t2)*4
        
        # Load element from original matrix
        flw ft0, 0(t4)
        
        # Calculate address in transposed matrix: a0 + (t2 + 8*t3)*4
        # For transposed matrix (8x144), element at (col t2, row t3) is at a0 + (t2 + 8*t3)*4
        li t5, 8
        mul t5, t5, t3     # t5 = 8*t3
        add t5, t5, t2     # t5 = t2 + 8*t3
        slli t5, t5, 2     # t5 = (t2 + 8*t3)*4
        add t5, a0, t5     # t5 = a0 + (t2 + 8*t3)*4
        
        # Store element to transposed matrix
        fsw ft0, 0(t5)
        
        addi t3, t3, 1     # Increment row counter
        li t6, 144
        blt t3, t6, inner_loop1  # Continue inner loop if t3 < 144
        
        addi t2, t2, 1     # Increment column counter
        li t6, 8
        blt t2, t6, outer_loop1  # Continue outer loop if t2 < 8

        ret


## END YOU CODE HERE

# Function: print
# Logs values from array in a0 into registers v1 for debugging and output.
# Inputs:
#   - a0: Base address of array
#   - a1: Size of array i.e. number of elements to log ## Safeguard: Actually it represents the rows of a square matrix though we could remove mul a1, a1, a1 line to work with a 1D array
# Clobbers: t0,t1, t2,t3 ft0, ft1.
printToLogVectorized:        
    addi sp, sp, -4
    sw a0, 0(sp)

    li t0, 0x123                 # Pattern for help in python script
    li t0, 0x456                 # Pattern for help in python script
    mv a1, a1                   # moving size to get it from log 
    mul a1, a1, a1              # sqaure matrix has n^2 elements 
	addi t0, x0, 0		                # load i = 0
    printloop:
        vsetvli t3, a1, e32           # Set VLEN based on a1
        slli t4, t3, 2                # Compute VLEN * 4 for address increment

        vle32.v v1, (a0)              # Load real[i] into v1
        add a0, a0, t4                # Increment pointer for real[] by VLEN * 4
        add t0, t0, t3                # Increment index

        bge t0, a1, endPrintLoop      # Exit loop if i >= size
        j printloop                   # Jump to start of loop
    endPrintLoop:
    li t0, 0x123                    # Pattern for help in python script
    li t0, 0x456                    # Pattern for help in python script
	
    lw a0, 0(sp)
    addi sp, sp, 4
	jr ra #(from TA's original code so commented out)
    # Important (creating infinite loop during runtime)

# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr

## ALL DATA IS DEFINED HERE LIKE MATRIX, CONSTANTS ETC


## DATA DEFINE START
.equ MatrixSize, 10
matrix1:
    .float -111.75, -602.75, 646.0, -439.5, -25.75, 351.25, -699.0, 736.5, -69.5, -453.75
    .float -195.5, 254.5, -616.75, 712.5, 382.75, 532.25, 656.0, 309.0, -580.0, -91.75
    .float 793.25, -207.0, -893.25, -652.25, 66.25, -734.5, -522.0, 68.75, 894.75, -80.0
    .float 18.5, 953.5, 288.75, -235.0, 780.25, -577.0, -959.5, 723.25, -513.5, -909.75
    .float 310.0, 318.0, 973.75, 1.25, 443.5, 982.5, 265.75, -552.5, -273.0, 740.25
    .float -885.75, -964.5, 485.75, -134.75, -399.0, -374.0, 205.5, -62.75, 3.75, 442.5
    .float -23.25, 182.0, 845.0, -370.75, 450.5, -932.25, 779.0, -635.75, 571.75, 102.75
    .float 137.5, 568.25, -114.0, 813.0, 982.75, 698.0, 549.0, -291.0, 397.0, -961.5
    .float 861.5, 384.0, 454.0, 892.25, -412.0, 653.75, 850.75, 607.5, 791.75, 78.25
    .float -438.5, 378.5, 823.5, 938.25, -637.25, 390.5, -857.25, 790.5, 988.0, 357.0
## DATA DEFINE END
size1: .word MatrixSize

.bss
.align 2
image_patch_buffer:
    .space 100    # 25 floats × 4 bytes
output_max:
    .space 4608 # 12*12*8*4 #float

# conv_output: 
#     .space 8*24*24*4 # 8 channels, height and width 24x24 and word size 4 bytes


.section .data 
.align 2
test: .float 6.9

conv_output: 
    .space 18432 # 8*24*24*4 # 8 channels, height and width 24x24 and word size 4 bytes

output_max_flattened:
    .space 4608

p:
    .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

dense_outputs:
    .float 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

float_zero: .float 0.0

test_matrix:    # 24x24x8 matrix
    .float 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
    .float 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48
    .float 49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72
    .float 73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96
    .float 97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120
    .float 121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144
    .float 145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168
    .float 169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192
    .float 193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216
    .float 217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240
    .float 241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264
    .float 265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288
    .float 289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312
    .float 313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336
    .float 337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360
    .float 361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384
    .float 385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408
    .float 409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432
    .float 433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456
    .float 457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480
    .float 481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504
    .float 505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528
    .float 529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552
    .float 553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576
    .float 577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600
    .float 601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624
    .float 625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648
    .float 649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672
    .float 673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696
    .float 697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720
    .float 721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744
    .float 745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768
    .float 769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792
    .float 793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816
    .float 817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840
    .float 841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864
    .float 865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888
    .float 889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912
    .float 913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936
    .float 937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960
    .float 961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984
    .float 985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008
    .float 1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032
    .float 1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056
    .float 1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080
    .float 1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104
    .float 1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128
    .float 1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152
    .float 1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176
    .float 1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200
    .float 1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224
    .float 1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248
    .float 1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272
    .float 1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296
    .float 1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320
    .float 1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344
    .float 1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368
    .float 1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392
    .float 1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416
    .float 1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440
    .float 1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451,1452,1453,1454,1455,1456,1457,1458,1459,1460,1461,1462,1463,1464
    .float 1465,1466,1467,1468,1469,1470,1471,1472,1473,1474,1475,1476,1477,1478,1479,1480,1481,1482,1483,1484,1485,1486,1487,1488
    .float 1489,1490,1491,1492,1493,1494,1495,1496,1497,1498,1499,1500,1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512
    .float 1513,1514,1515,1516,1517,1518,1519,1520,1521,1522,1523,1524,1525,1526,1527,1528,1529,1530,1531,1532,1533,1534,1535,1536
    .float 1537,1538,1539,1540,1541,1542,1543,1544,1545,1546,1547,1548,1549,1550,1551,1552,1553,1554,1555,1556,1557,1558,1559,1560
    .float 1561,1562,1563,1564,1565,1566,1567,1568,1569,1570,1571,1572,1573,1574,1575,1576,1577,1578,1579,1580,1581,1582,1583,1584
    .float 1585,1586,1587,1588,1589,1590,1591,1592,1593,1594,1595,1596,1597,1598,1599,1600,1601,1602,1603,1604,1605,1606,1607,1608
    .float 1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632
    .float 1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1652,1653,1654,1655,1656
    .float 1657,1658,1659,1660,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680
    .float 1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704
    .float 1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728
    .float 1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1745,1746,1747,1748,1749,1750,1751,1752
    .float 1753,1754,1755,1756,1757,1758,1759,1760,1761,1762,1763,1764,1765,1766,1767,1768,1769,1770,1771,1772,1773,1774,1775,1776
    .float 1777,1778,1779,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799,1800
    .float 1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824
    .float 1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848
    .float 1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,1860,1861,1862,1863,1864,1865,1866,1867,1868,1869,1870,1871,1872
    .float 1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896
    .float 1897,1898,1899,1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920
    .float 1921,1922,1923,1924,1925,1926,1927,1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943,1944
    .float 1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968
    .float 1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992
    .float 1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016
    .float 2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040
    .float 2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,2060,2061,2062,2063,2064
    .float 2065,2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088
    .float 2089,2090,2091,2092,2093,2094,2095,2096,2097,2098,2099,2100,2101,2102,2103,2104,2105,2106,2107,2108,2109,2110,2111,2112
    .float 2113,2114,2115,2116,2117,2118,2119,2120,2121,2122,2123,2124,2125,2126,2127,2128,2129,2130,2131,2132,2133,2134,2135,2136
    .float 2137,2138,2139,2140,2141,2142,2143,2144,2145,2146,2147,2148,2149,2150,2151,2152,2153,2154,2155,2156,2157,2158,2159,2160
    .float 2161,2162,2163,2164,2165,2166,2167,2168,2169,2170,2171,2172,2173,2174,2175,2176,2177,2178,2179,2180,2181,2182,2183,2184
    .float 2185,2186,2187,2188,2189,2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2206,2207,2208
    .float 2209,2210,2211,2212,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2225,2226,2227,2228,2229,2230,2231,2232
    .float 2233,2234,2235,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2246,2247,2248,2249,2250,2251,2252,2253,2254,2255,2256
    .float 2257,2258,2259,2260,2261,2262,2263,2264,2265,2266,2267,2268,2269,2270,2271,2272,2273,2274,2275,2276,2277,2278,2279,2280
    .float 2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304
    .float 2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2327,2328
    .float 2329,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2346,2347,2348,2349,2350,2351,2352
    .float 2353,2354,2355,2356,2357,2358,2359,2360,2361,2362,2363,2364,2365,2366,2367,2368,2369,2370,2371,2372,2373,2374,2375,2376
    .float 2377,2378,2379,2380,2381,2382,2383,2384,2385,2386,2387,2388,2389,2390,2391,2392,2393,2394,2395,2396,2397,2398,2399,2400
    .float 2401,2402,2403,2404,2405,2406,2407,2408,2409,2410,2411,2412,2413,2414,2415,2416,2417,2418,2419,2420,2421,2422,2423,2424
    .float 2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2448
    .float 2449,2450,2451,2452,2453,2454,2455,2456,2457,2458,2459,2460,2461,2462,2463,2464,2465,2466,2467,2468,2469,2470,2471,2472
    .float 2473,2474,2475,2476,2477,2478,2479,2480,2481,2482,2483,2484,2485,2486,2487,2488,2489,2490,2491,2492,2493,2494,2495,2496
    .float 2497,2498,2499,2500,2501,2502,2503,2504,2505,2506,2507,2508,2509,2510,2511,2512,2513,2514,2515,2516,2517,2518,2519,2520
    .float 2521,2522,2523,2524,2525,2526,2527,2528,2529,2530,2531,2532,2533,2534,2535,2536,2537,2538,2539,2540,2541,2542,2543,2544
    .float 2545,2546,2547,2548,2549,2550,2551,2552,2553,2554,2555,2556,2557,2558,2559,2560,2561,2562,2563,2564,2565,2566,2567,2568
    .float 2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589,2590,2591,2592
    .float 2593,2594,2595,2596,2597,2598,2599,2600,2601,2602,2603,2604,2605,2606,2607,2608,2609,2610,2611,2612,2613,2614,2615,2616
    .float 2617,2618,2619,2620,2621,2622,2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2633,2634,2635,2636,2637,2638,2639,2640
    .float 2641,2642,2643,2644,2645,2646,2647,2648,2649,2650,2651,2652,2653,2654,2655,2656,2657,2658,2659,2660,2661,2662,2663,2664
    .float 2665,2666,2667,2668,2669,2670,2671,2672,2673,2674,2675,2676,2677,2678,2679,2680,2681,2682,2683,2684,2685,2686,2687,2688
    .float 2689,2690,2691,2692,2693,2694,2695,2696,2697,2698,2699,2700,2701,2702,2703,2704,2705,2706,2707,2708,2709,2710,2711,2712
    .float 2713,2714,2715,2716,2717,2718,2719,2720,2721,2722,2723,2724,2725,2726,2727,2728,2729,2730,2731,2732,2733,2734,2735,2736
    .float 2737,2738,2739,2740,2741,2742,2743,2744,2745,2746,2747,2748,2749,2750,2751,2752,2753,2754,2755,2756,2757,2758,2759,2760
    .float 2761,2762,2763,2764,2765,2766,2767,2768,2769,2770,2771,2772,2773,2774,2775,2776,2777,2778,2779,2780,2781,2782,2783,2784
    .float 2785,2786,2787,2788,2789,2790,2791,2792,2793,2794,2795,2796,2797,2798,2799,2800,2801,2802,2803,2804,2805,2806,2807,2808
    .float 2809,2810,2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828,2829,2830,2831,2832
    .float 2833,2834,2835,2836,2837,2838,2839,2840,2841,2842,2843,2844,2845,2846,2847,2848,2849,2850,2851,2852,2853,2854,2855,2856
    .float 2857,2858,2859,2860,2861,2862,2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2873,2874,2875,2876,2877,2878,2879,2880
    .float 2881,2882,2883,2884,2885,2886,2887,2888,2889,2890,2891,2892,2893,2894,2895,2896,2897,2898,2899,2900,2901,2902,2903,2904
    .float 2905,2906,2907,2908,2909,2910,2911,2912,2913,2914,2915,2916,2917,2918,2919,2920,2921,2922,2923,2924,2925,2926,2927,2928
    .float 2929,2930,2931,2932,2933,2934,2935,2936,2937,2938,2939,2940,2941,2942,2943,2944,2945,2946,2947,2948,2949,2950,2951,2952
    .float 2953,2954,2955,2956,2957,2958,2959,2960,2961,2962,2963,2964,2965,2966,2967,2968,2969,2970,2971,2972,2973,2974,2975,2976
    .float 2977,2978,2979,2980,2981,2982,2983,2984,2985,2986,2987,2988,2989,2990,2991,2992,2993,2994,2995,2996,2997,2998,2999,3000
    .float 3001,3002,3003,3004,3005,3006,3007,3008,3009,3010,3011,3012,3013,3014,3015,3016,3017,3018,3019,3020,3021,3022,3023,3024
    .float 3025,3026,3027,3028,3029,3030,3031,3032,3033,3034,3035,3036,3037,3038,3039,3040,3041,3042,3043,3044,3045,3046,3047,3048
    .float 3049,3050,3051,3052,3053,3054,3055,3056,3057,3058,3059,3060,3061,3062,3063,3064,3065,3066,3067,3068,3069,3070,3071,3072
    .float 3073,3074,3075,3076,3077,3078,3079,3080,3081,3082,3083,3084,3085,3086,3087,3088,3089,3090,3091,3092,3093,3094,3095,3096
    .float 3097,3098,3099,3100,3101,3102,3103,3104,3105,3106,3107,3108,3109,3110,3111,3112,3113,3114,3115,3116,3117,3118,3119,3120
    .float 3121,3122,3123,3124,3125,3126,3127,3128,3129,3130,3131,3132,3133,3134,3135,3136,3137,3138,3139,3140,3141,3142,3143,3144
    .float 3145,3146,3147,3148,3149,3150,3151,3152,3153,3154,3155,3156,3157,3158,3159,3160,3161,3162,3163,3164,3165,3166,3167,3168
    .float 3169,3170,3171,3172,3173,3174,3175,3176,3177,3178,3179,3180,3181,3182,3183,3184,3185,3186,3187,3188,3189,3190,3191,3192
    .float 3193,3194,3195,3196,3197,3198,3199,3200,3201,3202,3203,3204,3205,3206,3207,3208,3209,3210,3211,3212,3213,3214,3215,3216
    .float 3217,3218,3219,3220,3221,3222,3223,3224,3225,3226,3227,3228,3229,3230,3231,3232,3233,3234,3235,3236,3237,3238,3239,3240
    .float 3241,3242,3243,3244,3245,3246,3247,3248,3249,3250,3251,3252,3253,3254,3255,3256,3257,3258,3259,3260,3261,3262,3263,3264
    .float 3265,3266,3267,3268,3269,3270,3271,3272,3273,3274,3275,3276,3277,3278,3279,3280,3281,3282,3283,3284,3285,3286,3287,3288
    .float 3289,3290,3291,3292,3293,3294,3295,3296,3297,3298,3299,3300,3301,3302,3303,3304,3305,3306,3307,3308,3309,3310,3311,3312
    .float 3313,3314,3315,3316,3317,3318,3319,3320,3321,3322,3323,3324,3325,3326,3327,3328,3329,3330,3331,3332,3333,3334,3335,3336
    .float 3337,3338,3339,3340,3341,3342,3343,3344,3345,3346,3347,3348,3349,3350,3351,3352,3353,3354,3355,3356,3357,3358,3359,3360
    .float 3361,3362,3363,3364,3365,3366,3367,3368,3369,3370,3371,3372,3373,3374,3375,3376,3377,3378,3379,3380,3381,3382,3383,3384
    .float 3385,3386,3387,3388,3389,3390,3391,3392,3393,3394,3395,3396,3397,3398,3399,3400,3401,3402,3403,3404,3405,3406,3407,3408
    .float 3409,3410,3411,3412,3413,3414,3415,3416,3417,3418,3419,3420,3421,3422,3423,3424,3425,3426,3427,3428,3429,3430,3431,3432
    .float 3433,3434,3435,3436,3437,3438,3439,3440,3441,3442,3443,3444,3445,3446,3447,3448,3449,3450,3451,3452,3453,3454,3455,3456
    .float 3457,3458,3459,3460,3461,3462,3463,3464,3465,3466,3467,3468,3469,3470,3471,3472,3473,3474,3475,3476,3477,3478,3479,3480
    .float 3481,3482,3483,3484,3485,3486,3487,3488,3489,3490,3491,3492,3493,3494,3495,3496,3497,3498,3499,3500,3501,3502,3503,3504
    .float 3505,3506,3507,3508,3509,3510,3511,3512,3513,3514,3515,3516,3517,3518,3519,3520,3521,3522,3523,3524,3525,3526,3527,3528
    .float 3529,3530,3531,3532,3533,3534,3535,3536,3537,3538,3539,3540,3541,3542,3543,3544,3545,3546,3547,3548,3549,3550,3551,3552
    .float 3553,3554,3555,3556,3557,3558,3559,3560,3561,3562,3563,3564,3565,3566,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576
    .float 3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600
    .float 3601,3602,3603,3604,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624
    .float 3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,3646,3647,3648
    .float 3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672
    .float 3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696
    .float 3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720
    .float 3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744
    .float 3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768
    .float 3769,3770,3771,3772,3773,3774,3775,3776,3777,3778,3779,3780,3781,3782,3783,3784,3785,3786,3787,3788,3789,3790,3791,3792
    .float 3793,3794,3795,3796,3797,3798,3799,3800,3801,3802,3803,3804,3805,3806,3807,3808,3809,3810,3811,3812,3813,3814,3815,3816
    .float 3817,3818,3819,3820,3821,3822,3823,3824,3825,3826,3827,3828,3829,3830,3831,3832,3833,3834,3835,3836,3837,3838,3839,3840
    .float 3841,3842,3843,3844,3845,3846,3847,3848,3849,3850,3851,3852,3853,3854,3855,3856,3857,3858,3859,3860,3861,3862,3863,3864
    .float 3865,3866,3867,3868,3869,3870,3871,3872,3873,3874,3875,3876,3877,3878,3879,3880,3881,3882,3883,3884,3885,3886,3887,3888
    .float 3889,3890,3891,3892,3893,3894,3895,3896,3897,3898,3899,3900,3901,3902,3903,3904,3905,3906,3907,3908,3909,3910,3911,3912
    .float 3913,3914,3915,3916,3917,3918,3919,3920,3921,3922,3923,3924,3925,3926,3927,3928,3929,3930,3931,3932,3933,3934,3935,3936
    .float 3937,3938,3939,3940,3941,3942,3943,3944,3945,3946,3947,3948,3949,3950,3951,3952,3953,3954,3955,3956,3957,3958,3959,3960
    .float 3961,3962,3963,3964,3965,3966,3967,3968,3969,3970,3971,3972,3973,3974,3975,3976,3977,3978,3979,3980,3981,3982,3983,3984
    .float 3985,3986,3987,3988,3989,3990,3991,3992,3993,3994,3995,3996,3997,3998,3999,4000,4001,4002,4003,4004,4005,4006,4007,4008
    .float 4009,4010,4011,4012,4013,4014,4015,4016,4017,4018,4019,4020,4021,4022,4023,4024,4025,4026,4027,4028,4029,4030,4031,4032
    .float 4033,4034,4035,4036,4037,4038,4039,4040,4041,4042,4043,4044,4045,4046,4047,4048,4049,4050,4051,4052,4053,4054,4055,4056
    .float 4057,4058,4059,4060,4061,4062,4063,4064,4065,4066,4067,4068,4069,4070,4071,4072,4073,4074,4075,4076,4077,4078,4079,4080
    .float 4081,4082,4083,4084,4085,4086,4087,4088,4089,4090,4091,4092,4093,4094,4095,4096,4097,4098,4099,4100,4101,4102,4103,4104
    .float 4105,4106,4107,4108,4109,4110,4111,4112,4113,4114,4115,4116,4117,4118,4119,4120,4121,4122,4123,4124,4125,4126,4127,4128
    .float 4129,4130,4131,4132,4133,4134,4135,4136,4137,4138,4139,4140,4141,4142,4143,4144,4145,4146,4147,4148,4149,4150,4151,4152
    .float 4153,4154,4155,4156,4157,4158,4159,4160,4161,4162,4163,4164,4165,4166,4167,4168,4169,4170,4171,4172,4173,4174,4175,4176
    .float 4177,4178,4179,4180,4181,4182,4183,4184,4185,4186,4187,4188,4189,4190,4191,4192,4193,4194,4195,4196,4197,4198,4199,4200
    .float 4201,4202,4203,4204,4205,4206,4207,4208,4209,4210,4211,4212,4213,4214,4215,4216,4217,4218,4219,4220,4221,4222,4223,4224
    .float 4225,4226,4227,4228,4229,4230,4231,4232,4233,4234,4235,4236,4237,4238,4239,4240,4241,4242,4243,4244,4245,4246,4247,4248
    .float 4249,4250,4251,4252,4253,4254,4255,4256,4257,4258,4259,4260,4261,4262,4263,4264,4265,4266,4267,4268,4269,4270,4271,4272
    .float 4273,4274,4275,4276,4277,4278,4279,4280,4281,4282,4283,4284,4285,4286,4287,4288,4289,4290,4291,4292,4293,4294,4295,4296
    .float 4297,4298,4299,4300,4301,4302,4303,4304,4305,4306,4307,4308,4309,4310,4311,4312,4313,4314,4315,4316,4317,4318,4319,4320
    .float 4321,4322,4323,4324,4325,4326,4327,4328,4329,4330,4331,4332,4333,4334,4335,4336,4337,4338,4339,4340,4341,4342,4343,4344
    .float 4345,4346,4347,4348,4349,4350,4351,4352,4353,4354,4355,4356,4357,4358,4359,4360,4361,4362,4363,4364,4365,4366,4367,4368
    .float 4369,4370,4371,4372,4373,4374,4375,4376,4377,4378,4379,4380,4381,4382,4383,4384,4385,4386,4387,4388,4389,4390,4391,4392
    .float 4393,4394,4395,4396,4397,4398,4399,4400,4401,4402,4403,4404,4405,4406,4407,4408,4409,4410,4411,4412,4413,4414,4415,4416
    .float 4417,4418,4419,4420,4421,4422,4423,4424,4425,4426,4427,4428,4429,4430,4431,4432,4433,4434,4435,4436,4437,4438,4439,4440
    .float 4441,4442,4443,4444,4445,4446,4447,4448,4449,4450,4451,4452,4453,4454,4455,4456,4457,4458,4459,4460,4461,4462,4463,4464
    .float 4465,4466,4467,4468,4469,4470,4471,4472,4473,4474,4475,4476,4477,4478,4479,4480,4481,4482,4483,4484,4485,4486,4487,4488
    .float 4489,4490,4491,4492,4493,4494,4495,4496,4497,4498,4499,4500,4501,4502,4503,4504,4505,4506,4507,4508,4509,4510,4511,4512
    .float 4513,4514,4515,4516,4517,4518,4519,4520,4521,4522,4523,4524,4525,4526,4527,4528,4529,4530,4531,4532,4533,4534,4535,4536
    .float 4537,4538,4539,4540,4541,4542,4543,4544,4545,4546,4547,4548,4549,4550,4551,4552,4553,4554,4555,4556,4557,4558,4559,4560
    .float 4561,4562,4563,4564,4565,4566,4567,4568,4569,4570,4571,4572,4573,4574,4575,4576,4577,4578,4579,4580,4581,4582,4583,4584
    .float 4585,4586,4587,4588,4589,4590,4591,4592,4593,4594,4595,4596,4597,4598,4599,4600,4601,4602,4603,4604,4605,4606,4607,4608

## Fully Connected Layer Weights and Biases, All data based on a similar tensorflow CNN with 97% accuracy trained on unaugmented but normalized MNIST data 
#.global weights
# 1152 x 10 Dense weights

## DENSE_WEIGHTS BEGIN
dense_weights:
    .float 0.08019345, -0.19868167, -0.01696709, -0.07749116, -0.13918322, -0.01632601, -0.14505003, -0.10418198, -0.10674403, -0.09506100, -0.02308065, -0.11408407, -0.07840892, -0.06742292, 0.01832916, 0.01026583, -0.16590583, -0.02510842, -0.00990188, -0.11978007, -0.08418796, -0.10584393, 0.01404687, -0.00675779, 0.02232841, -0.07126692, -0.07569554, -0.07387549, -0.04616801, -0.09340189, 0.05345195, -0.07304040, -0.06463636, 0.02680550, -0.00573927, 0.04448839, -0.09519269, 0.00115245, -0.04911669, -0.11933945, -0.04698307, -0.08815598, 0.08505419, 0.11278614, -0.09149384, 0.01123283, -0.04404208, 0.00186711, -0.03711207, -0.10889626, 0.00172118, -0.01048115, -0.03203008, 0.00524875, -0.07925984, 0.00757882, 0.03447837, -0.06792871, -0.06774517, 0.02449953, -0.09306407, 0.03794711, -0.05346283, -0.07513785, -0.07903468, -0.13099790, -0.01448584, -0.05207442, -0.08490730, 0.00260947, -0.11085509, -0.07794001, -0.06271214, -0.11020306, -0.05437456, -0.01725841, 0.02193302, 0.02451329, -0.15328470, 0.01295777, -0.00229508, -0.03463621, -0.11683556, -0.01398881, -0.09050357, -0.06151027, -0.02751818, -0.02007603, -0.00681891, 0.01408735, -0.05388410, 0.01267488, -0.04567245, 0.00574604, 0.00462211, -0.02952557, -0.07034263, -0.08932512, -0.10124150, -0.00160356, -0.10712840, -0.05448998, -0.06571057, -0.01517161, -0.13153020, 0.01024853, -0.01542290, -0.02320928, -0.09344802, -0.06992777, -0.03782647, 0.01204330, -0.02952523, -0.01439001, -0.02319661, -0.07328705, -0.06701675, -0.04775646, 0.02656374, 0.04405220, -0.09182314, -0.03806709, 0.00021552, -0.00000052, 0.02989722, -0.13211982, 0.05638721, -0.06552760, -0.10914835, -0.02383845, 0.03209848, -0.04194475, -0.00200518, -0.19070955, -0.05669945, -0.07409420, -0.02059447, -0.01915442, -0.02436726, -0.05710253, 0.06547932, -0.11083483, 0.04204230, 0.01827495, -0.06001200, 0.07290473, -0.03687902, 0.02298523, 0.04080208, -0.16736509, -0.04536984, 0.01631149, 0.03749168, -0.04293824, 0.01315535, -0.05358090, -0.01837287, 0.07230057, 0.04380631, -0.05402489, 0.01937545, -0.05058599, -0.12345074, -0.00035120, -0.04497787, 0.01516508, -0.08341993, 0.00119890, -0.05400798, -0.12813838, -0.16438958, -0.10819898, -0.05004251, -0.09265783, 0.04250237, 0.00635894, 0.02431570, -0.00072116, -0.09343936, 0.01803946, 0.05337513, -0.11238885, 0.01307648, -0.00051548, -0.01913960, -0.15708476, -0.16747510, 0.08419249, 0.11528079, -0.05468426, -0.00368670, -0.08660325, -0.02265003, -0.03651769, -0.04752169, 0.00030942, -0.10527006, -0.12162900, 0.01533842, 0.05445129, -0.07517044, 0.05152693, -0.01983120, -0.01230731, 0.01322347, -0.06809078, 0.05632122, 0.06038816, -0.00421349, -0.03037200, -0.05687034, -0.14206180, 0.00637962, -0.15322536, 0.03742374, 0.03140428, -0.10174943, 0.04062587, -0.05199069, -0.11163826, 0.03915919, -0.06541841, 0.04358460, -0.05801779, 0.02087360, -0.07945536, 0.00495420, -0.05160623, 0.00097498, -0.17488590, 0.01275958, -0.03838832, -0.06283277, -0.07237180, -0.05231095, -0.02380466, 0.01835953, -0.21263012, -0.01710964, 0.02380748, -0.04798982, -0.01269111, -0.01213087, -0.05929321, -0.03656106, -0.08552484, 0.05183342, 0.05026326, 0.01757399, 0.01307137, -0.13067806, -0.12019113, -0.03180280, -0.01825826, 0.04707335, 0.05660558, -0.07511574, 0.02882186, -0.20460159, 0.00504201, -0.03144372, -0.01839853, 0.09468409, 0.00766215, -0.02785359, -0.03172875, -0.21845470, 0.04261123, 0.04022033, -0.05945378, -0.04821711, -0.04739980, 0.02824646, -0.01701863, -0.17710464, 0.05944493, 0.01602829, -0.04797016, 0.08015474, -0.00875564, -0.02861513, -0.04685104, -0.24828565, 0.03823498, 0.03962537, -0.10626098, 0.02437234, -0.00854858, 0.02720114, 0.06657755, -0.03228419, -0.04009741, -0.03332742, 0.00860741, -0.05352132, -0.05412489, -0.03682670, -0.06236709, 0.04184564, 0.00298291, -0.04736967, -0.03923494, -0.02758991, -0.02529438, 0.00356893, -0.10760650, -0.06868026, -0.03484602, 0.05427787, -0.02652749, -0.04253162, 0.02778694, 0.02090950, -0.07114326, 0.01982614, 0.00590204, 0.04904638, -0.01857405, -0.06865482, -0.05442681, -0.00543472, -0.06076624, -0.08826213, 0.06869753, -0.04632884, -0.02230270, -0.05807995, -0.04311196, 0.02448741, -0.08384324, -0.02610516, 0.06743681, 0.08818355, -0.00681598, 0.02210769, 0.01460380, 0.03932285, 0.03288212, 0.01676084, -0.07620411, 0.10292381, -0.07120681, 0.02122981, 0.01221991, -0.10243937, 0.05457263, -0.24966581, -0.03757368, 0.03771634, 0.00987404, 0.03467660, 0.02077636, -0.15660767, 0.12796144, -0.28481692, -0.00354334, 0.04131762, 0.06785565, 0.08448137, 0.05207533, -0.06818797, 0.01704405, -0.32424897, 0.06033483, 0.11868694, -0.00474833, 0.05170587, 0.06778326, -0.01353493, 0.00725844, -0.32410040, 0.01352121, 0.04734941, 0.08234236, -0.00156551, -0.01105986, -0.12062649, -0.10549387, -0.13155904, 0.12294505, 0.01488760, -0.05432563, 0.01926162, -0.01513250, -0.01073481, -0.05786844, 0.04107648, -0.00888975, -0.04711644, -0.00264656, 0.03306857, -0.10034800, -0.09173108, -0.01233417, 0.00290216, 0.05716509, 0.05287398, 0.00833330, 0.05943862, 0.02040045, 0.02032042, -0.08006442, 0.02407245, 0.14012046, -0.04323358, -0.04377338, 0.02374861, 0.02403048, 0.05403720, -0.02042404, 0.02073515, 0.07888325, 0.08147494, 0.00486683, -0.04847886, -0.00607286, -0.02446077, -0.03228988, 0.04361813, 0.12127591, 0.09348916, 0.01716844, 0.00466125, -0.03029094, 0.00787663, -0.13019364, -0.08503600, 0.11499145, -0.02088649, -0.06401194, 0.04409143, 0.02333665, -0.16482292, -0.07251073, -0.27213937, 0.10214759, 0.00888219, -0.07349499, -0.01898745, 0.02473796, -0.15155116, 0.12029794, -0.33303970, 0.01170218, 0.04178092, 0.12700154, -0.01209973, 0.10041749, -0.22898537, 0.12407900, -0.27112794, 0.06039904, -0.01690712, 0.09224334, -0.06189017, 0.05690765, -0.12221929, 0.02661735, -0.20233773, 0.13255852, 0.05891224, -0.01717847, 0.02383901, 0.13194217, -0.13069354, 0.03956364, -0.23716082, 0.11202113, 0.01204325, 0.03671581, 0.13284314, 0.06587895, -0.11252920, -0.02462104, -0.24026237, 0.19749862, 0.12304193, -0.11074634, 0.09648374, -0.03757670, -0.04438870, -0.05611373, -0.04333328, 0.06996513, -0.05330429, -0.13414907, 0.09586184, -0.01389182, -0.02647530, -0.06202084, 0.06240996, 0.09738988, -0.05479242, -0.06049331, 0.05265274, -0.05520310, 0.09481664, -0.08627486, 0.03684828, 0.13846871, -0.04695217, -0.04055745, 0.04124364, 0.03642566, 0.01922927, -0.01198864, 0.02400924, 0.22017343, -0.00084074, -0.09580718, -0.03737084, 0.03206796, 0.04282656, -0.13718477, 0.07626992, 0.13642308, -0.05301751, -0.10523947, 0.10751305, -0.01719536, -0.07751756, -0.20193528, -0.20467988, 0.18644933, -0.11351269, -0.22521365, 0.06624192, -0.09169617, -0.17810974, -0.17585827, -0.36472338, 0.19556952, -0.09951746, -0.13829757, -0.06823069, -0.11369178, -0.04493879, -0.09706123, -0.20509233, 0.02506160, -0.12324315, 0.04942918, -0.01221927, -0.03729434, -0.05807487, -0.09545878, -0.04747479, 0.10082977, 0.05148639, 0.06358354, 0.02387318, -0.00155986, -0.08892621, 0.06097305, -0.03210837, 0.12770441, 0.12806743, 0.00124053, -0.03436127, 0.03194427, -0.13369611, 0.11966605, -0.12419930, 0.07365885, 0.03887824, 0.00776316, 0.12610981, 0.10440445, -0.02792714, 0.05658333, -0.23107247, 0.19792272, 0.14826378, -0.16795468, 0.06772505, 0.07376830, -0.03781369, 0.01753493, 0.05588083, 0.01794280, -0.03436647, -0.10054533, 0.03660092, 0.03907952, 0.04150849, -0.00612917, -0.05845307, 0.02085418, 0.05627360, -0.03471314, -0.06593817, 0.04555064, 0.10688504, 0.03109732, -0.00256585, 0.16111910, 0.07487051, -0.02694524, -0.00258312, 0.07444680, 0.08234785, -0.02796093, -0.02039839, 0.16496742, 0.04564115, -0.11579426, 0.10582554, 0.11000312, -0.01821182, -0.08823859, -0.12583145, 0.08680857, -0.01735215, -0.00285718, 0.06472497, -0.01705156, -0.09924032, -0.11647707, -0.31339839, 0.29729223, -0.09263876, -0.26639533, -0.00990774, -0.13904701, -0.17561392, -0.09440088, -0.14146429, 0.15786719, -0.08140910, -0.17224117, 0.00351518, -0.14641915, -0.06310873, -0.17619151, -0.01753071, 0.02528881, -0.05237789, -0.12268312, -0.04356388, -0.08035880, 0.02615130, -0.09907404, -0.08753031, 0.06960766, 0.04589913, -0.00807884, 0.02141193, -0.00247905, -0.01531762, 0.01912806, 0.15523131, 0.05811416, 0.10327935, -0.04986042, -0.00122489, 0.04543741, 0.04098235, -0.01107829, -0.05188001, 0.08436559, 0.05417057, 0.01400034, 0.10100242, 0.10226291, -0.13438451, 0.02526366, -0.20646380, 0.04024492, 0.12301894, -0.07229884, 0.14963430, 0.10449864, 0.00055052, 0.06783159, 0.06325552, -0.03047141, -0.03579645, -0.08776502, 0.03064393, 0.02791203, 0.00565167, 0.01349334, -0.05651007, 0.09614835, 0.00524509, -0.01524848, -0.01589154, 0.01497558, 0.02600708, 0.02657577, -0.15301278, 0.15258425, 0.03645701, -0.03327868, 0.02256312, -0.03500626, -0.01030315, -0.03847663, -0.10746500, 0.17357284, -0.05663247, -0.05158825, 0.05373948, 0.07009123, 0.01587746, -0.07980485, -0.20488018, 0.10866616, 0.02100929, -0.07071025, 0.11935723, -0.00280315, -0.13574268, -0.06319270, -0.09173154, 0.29982433, -0.05796749, -0.19692230, 0.04797123, -0.00166505, 0.01540114, -0.15062007, 0.05947667, 0.13179296, -0.04492162, -0.19175984, -0.03612847, -0.13115084, 0.09300487, -0.10105949, 0.01356575, 0.08350080, -0.08957991, -0.15734248, -0.00039719, -0.05671901, -0.00714532, -0.04947955, 0.05098386, 0.11645858, 0.00456101, -0.12112030, 0.04308478, -0.00725509, 0.03090319, 0.02724840, 0.02610222, 0.04435528, 0.09651811, -0.04478873, 0.08953467, 0.09162044, 0.07986084, -0.02371817, -0.08207384, -0.00955406, 0.05048265, -0.03623182, -0.03397188, 0.09128603, -0.15155816, -0.01727749, -0.14210171, 0.01469744, 0.09348689, -0.04712392, 0.00988533, -0.04292408, -0.10317711, 0.05209508, -0.01007676, 0.02958997, 0.11154600, -0.01261531, 0.04084747, 0.00726889, -0.11623402, -0.01415870, -0.18800920, -0.00987873, 0.04208398, -0.07063287, -0.04658959, -0.01989817, 0.02075665, -0.02164588, -0.21808162, 0.05417451, 0.07578461, 0.00110370, -0.03759818, 0.06093781, -0.01438818, 0.06785258, -0.08863792, 0.10650396, -0.02609106, 0.06318543, 0.02562196, 0.08301704, 0.01879129, 0.09179772, -0.05139741, 0.11099898, -0.04380691, 0.06957034, 0.04577052, -0.03635031, 0.06763334, -0.09671673, -0.06118817, 0.10411150, -0.06190140, -0.19722337, 0.05487160, -0.09255467, 0.04027899, -0.14310896, 0.06891795, -0.01409637, -0.01291116, -0.11991852, 0.01896133, -0.04685301, 0.04074837, -0.01114698, 0.06689285, 0.09107009, 0.03667322, -0.00724350, -0.07904740, 0.00381373, -0.00867387, 0.01340043, 0.16228580, 0.05649970, 0.08063247, -0.00808811, 0.02198835, 0.08576302, 0.02281964, 0.02125221, 0.10154887, -0.03491585, -0.03316425, -0.08514021, 0.05053395, 0.05578117, -0.02897596, 0.03162628, 0.01835458, 0.01666683, 0.01914072, -0.07849117, 0.01717174, 0.06424858, -0.18557493, -0.01740407, -0.01540845, 0.00186656, 0.02949019, -0.05157888, 0.08781578, -0.05661468, 0.03185700, 0.02653283, -0.02969462, 0.00095355, 0.00719219, -0.09356688, -0.03405553, 0.07633794, -0.03648736, 0.08437693, -0.17088333, -0.11229295, -0.01791188, -0.03922978, 0.06086208, -0.01131365, -0.03358041, -0.02142191, -0.18601312, 0.02798886, 0.08238345, -0.05074900, -0.01033978, 0.04040466, 0.00453645, 0.03640036, -0.20968859, 0.11384635, 0.10598097, -0.02458318, -0.01359612, 0.00229881, 0.02906430, 0.01265690, -0.15826772, 0.05986499, 0.07758422, 0.13978548, 0.09783123, 0.07309666, 0.07362492, 0.07642327, -0.17677160, -0.05330261, 0.03614273, 0.03903699, -0.02249344, 0.00883705, 0.01000906, -0.05888588, -0.01027137, -0.13307995, 0.07991333, -0.05007523, -0.00753294, -0.07534320, 0.08833230, -0.06448955, 0.06128168, -0.04709829, 0.00318458, -0.14599441, -0.05687397, 0.00254752, -0.01423889, -0.07247772, 0.08887170, -0.05502869, 0.05485581, -0.08222792, -0.02754896, -0.06952282, 0.06068247, -0.04464638, 0.11764039, -0.11560909, -0.01194376, -0.06650996, -0.07905954, -0.08007230, 0.03473438, 0.02000949, -0.01982252, 0.01876082, -0.04896686, -0.07778357, -0.06504260, -0.02324672, -0.02211220, -0.02021311, -0.00077348, -0.05690366, -0.00612131, -0.06785561, -0.06119020, -0.10632558, 0.09286943, -0.05501218, -0.05290461, 0.01252636, 0.00455504, 0.00582309, 0.00769031, 0.05601653, 0.05620894, -0.01073487, -0.22429886, -0.01230717, 0.07658211, -0.02564953, -0.02124134, 0.13273719, -0.02109161, 0.04027634, -0.37070960, 0.00246504, -0.01860696, -0.02872670, 0.03895015, 0.00957607, 0.04697398, 0.13207167, -0.37383595, -0.12411610, 0.06805952, -0.01409212, 0.03384937, 0.01806276, 0.01107423, 0.05633101, -0.33522969, -0.14306603, 0.03497715, 0.13098544, 0.13759363, 0.08996075, -0.05007977, 0.07612137, -0.09550192, -0.15355258, 0.09737957, 0.05321421, 0.14690520, 0.02119339, 0.02386886, 0.12781896, -0.01219235, -0.14790672, -0.02701782, -0.04543319, 0.02190362, -0.03478685, 0.01686170, -0.05448617, 0.03956101, -0.21769880, -0.06423192, -0.17331746, -0.00512219, -0.02510514, -0.00218332, -0.11309652, -0.01667920, -0.15660352, -0.07050724, -0.11282952, -0.09391324, -0.12011167, -0.01267784, -0.04171958, 0.08245695, -0.11536141, -0.01328863, -0.16309531, -0.06363518, -0.05614883, -0.08693229, 0.01091730, 0.02129829, -0.04623637, -0.05197778, -0.11822613, -0.04128058, -0.11707082, 0.00770280, -0.11357813, -0.01800233, -0.11451510, -0.12485328, -0.11140148, -0.00061272, -0.15185684, -0.08815527, -0.03006700, -0.08121225, -0.09700177, 0.00617582, -0.07887435, 0.05081082, -0.02912349, 0.01503280, -0.12124094, -0.05791520, -0.10471555, 0.06938352, -0.06523382, 0.05781534, -0.01318031, 0.05003755, -0.00808899, -0.20566709, -0.12837741, 0.10499491, 0.08972365, 0.02388131, 0.02837964, 0.00113364, 0.04434169, -0.27129751, 0.02713018, 0.04635378, 0.05134505, 0.02431529, 0.04459365, -0.09646955, 0.01301617, -0.36015198, -0.03447407, 0.01596428, 0.15977055, 0.04676969, 0.09070228, -0.24258997, 0.09583872, -0.33626851, -0.17538635, -0.04558457, 0.13009472, 0.01499923, -0.00102924, -0.34586892, -0.04651187, -0.25730374, -0.12324528, -0.06161238, 0.05840586, -0.00420197, 0.04625507, -0.23715951, 0.01219402, -0.02136143, -0.10106924, -0.10349564, 0.06059529, -0.05402837, -0.08435488, -0.13995235, -0.15918387, -0.02840352, -0.09502456, -0.19303487, -0.00839503, -0.05339840, -0.08589048, -0.08510085, -0.16522409, -0.03311391, -0.18655586, -0.06223413, -0.01563751, -0.10981256, -0.18221447, -0.10683349, -0.11891268, -0.01497055, -0.20238961, -0.17578661, -0.09283423, -0.08732199, -0.08048593, 0.05316861, -0.03346600, 0.04555629, -0.04562969, -0.12746973, -0.08084250, -0.13853282, -0.18281940
    .float 0.13523541, 0.12751564, 0.11182610, 0.02401896, 0.12574144, 0.09447238, 0.05620605, 0.12641604, 0.02715605, 0.08419815, 0.01093775, -0.03575896, -0.06230952, 0.04363407, 0.09904015, -0.04172142, -0.00354233, -0.08948195, -0.07880048, 0.04903142, -0.03609075, -0.04783125, -0.05656838, -0.03180843, -0.03610396, -0.08136499, -0.05423544, 0.14301863, -0.20814653, -0.02218664, -0.04840131, 0.00930251, -0.08503321, 0.03143067, -0.07450136, 0.19369191, -0.06880436, 0.10621449, 0.02619182, -0.00673664, -0.04277803, -0.03772855, 0.04362025, 0.11103599, -0.10543270, 0.13252394, -0.00923776, 0.03503207, 0.03479941, -0.10676324, 0.00624080, 0.08042081, -0.02992355, 0.05834400, -0.04637671, 0.08619116, 0.08701911, -0.03446977, 0.04282027, 0.14382587, 0.12523347, 0.07076112, 0.01053627, 0.09140205, -0.01683101, 0.11935257, 0.06064409, 0.12182163, -0.02487479, 0.09710051, 0.02041179, 0.05137412, 0.13708068, -0.07341655, 0.06141783, 0.07049312, 0.08860236, 0.05302253, -0.04769392, 0.06789482, 0.01166451, -0.14465111, 0.01539553, 0.05450451, -0.00282594, 0.01960335, 0.04558687, -0.01219586, -0.04256134, -0.12091815, -0.07336427, 0.03224577, 0.00712714, 0.09195613, 0.03310577, 0.05122942, 0.00585733, 0.03639946, 0.01882359, 0.05585785, 0.15570743, 0.01813718, -0.01098649, 0.01063795, 0.00792752, 0.05917782, 0.01236550, -0.04286857, 0.02904225, 0.13572077, -0.10220625, -0.01492410, -0.05386819, 0.02075415, -0.03334184, -0.06580842, -0.07201521, 0.01024681, -0.01710742, -0.03116795, -0.20606899, 0.01191986, -0.08788360, 0.10080526, -0.07567696, -0.04142980, -0.10301849, -0.11186349, -0.17786051, 0.04474987, -0.13440642, 0.18925840, -0.04795129, -0.03845545, -0.20632604, -0.13604952, -0.09180096, -0.05703143, -0.02560787, 0.19521394, 0.02456416, -0.11441368, -0.05959987, -0.05624436, 0.00306484, -0.06848830, -0.02787990, 0.22325750, 0.00939729, -0.06378283, -0.04645231, -0.00028316, 0.04919990, -0.04999411, 0.04192627, 0.25033641, 0.01489668, 0.03973037, 0.04656050, -0.01939415, 0.01064453, -0.18006916, 0.12120498, 0.14198533, -0.05718849, 0.02637586, 0.02112976, 0.05533388, 0.01663714, -0.10866941, 0.14934707, 0.12936316, 0.06627858, -0.01543573, 0.06891949, 0.04518098, 0.06496993, -0.01676858, 0.00714389, 0.08468835, -0.05954269, -0.06596138, 0.00956325, 0.01942250, -0.08325683, -0.10921299, -0.06266549, 0.01039214, -0.03696459, 0.03575994, 0.05576647, -0.07622983, 0.08472193, -0.06515141, -0.06315634, 0.05769310, 0.01041112, 0.05133688, -0.04076827, -0.10321096, 0.02216429, -0.04970538, -0.12335814, -0.00452257, 0.02279161, 0.04600602, 0.01789876, 0.05778586, -0.12494550, -0.04933619, -0.10600846, 0.03931895, -0.13740455, 0.01777579, -0.05845694, -0.03080839, -0.18456696, -0.01731717, -0.10667273, 0.05513471, -0.01799612, -0.03199726, -0.11165307, -0.12016379, -0.13058864, -0.08382073, -0.14075148, 0.07482955, -0.03201659, -0.03815091, -0.12565750, -0.00203719, -0.09138997, -0.10178325, -0.10947300, 0.20553362, -0.07054967, -0.12888338, -0.13707404, 0.05372075, -0.11384539, -0.14003719, 0.08081105, 0.26682261, 0.02729577, -0.07368647, -0.00527060, 0.03812062, -0.13842240, -0.22257587, 0.09804285, 0.12755658, 0.09584717, -0.15399608, 0.07400342, -0.03329118, -0.09489037, -0.12642477, 0.10851412, 0.12196650, 0.07959032, -0.27341950, -0.04608678, 0.02129686, 0.01615298, -0.07655624, 0.16040848, 0.09067871, -0.01579053, -0.40770704, 0.12903868, -0.05451064, -0.07882473, 0.04055783, -0.13010174, 0.12926134, -0.04560320, -0.32202694, -0.01474213, -0.05720618, -0.21500093, -0.08239627, -0.03747970, -0.04178730, -0.13741697, -0.09512338, -0.03690742, -0.21590659, -0.09482445, -0.02912152, -0.01135301, 0.02507496, -0.10328215, -0.00655985, -0.14458133, -0.07936725, -0.12576784, -0.05975083, -0.03866262, 0.04862984, 0.02977795, 0.10088589, -0.07434000, -0.04564827, -0.11980922, -0.07739284, -0.04268204, -0.00111501, 0.00507634, 0.02108325, 0.00786978, -0.02524918, -0.02651163, 0.02706573, -0.09806909, -0.22436652, 0.01475695, -0.14209092, 0.01987893, 0.02928366, -0.15446575, -0.10883526, -0.11599490, -0.14244035, -0.10707124, -0.05672269, 0.00928911, -0.10114402, -0.05652126, -0.05229451, -0.12393343, -0.01673067, -0.04081934, -0.03772857, -0.08796844, 0.01274174, -0.12580529, -0.10909459, 0.09428928, 0.25128612, 0.06328024, -0.10814480, -0.05182646, 0.03610678, -0.06554462, -0.17187333, 0.22861010, 0.12554501, 0.02921794, -0.32083812, -0.02004688, -0.01564908, -0.15921396, -0.09126720, 0.16201515, 0.06981991, -0.05452494, -0.37482917, 0.03904757, -0.02024422, -0.02408847, -0.07510621, -0.10655352, 0.00813119, -0.07557356, -0.43891367, -0.05849112, -0.03329206, -0.18013513, -0.00690092, 0.00842965, -0.00736165, -0.00463946, -0.12930131, 0.05317744, -0.09754333, -0.19518767, -0.13350739, -0.05487398, 0.03875583, -0.12510841, -0.08747222, -0.09523755, -0.14738572, -0.11042380, -0.04730878, 0.05368775, -0.03583393, -0.03373090, 0.03920301, -0.02997449, 0.01622496, 0.01856418, -0.01765366, -0.08531948, 0.08012819, 0.01268928, 0.06734982, -0.06037586, -0.06063968, -0.06823954, 0.03330663, -0.09099090, 0.00359511, -0.06147548, 0.00781757, 0.02984725, -0.04497058, 0.01129868, -0.02742282, -0.09276072, -0.22882009, 0.01020635, -0.02536624, -0.13081901, -0.07862805, -0.09167061, -0.08392893, 0.00774505, -0.14262080, -0.05249774, -0.01538829, -0.04311796, -0.08884306, -0.05636578, 0.11956982, -0.03920212, -0.10207272, 0.05127358, -0.07274599, -0.04704709, 0.11230971, -0.17289366, -0.04802378, 0.20876679, 0.08967739, 0.09622295, 0.08051443, 0.07471005, 0.02847045, -0.22259797, -0.00753157, 0.02953736, 0.04340686, 0.06262083, -0.20023407, 0.11673118, -0.00215827, -0.29451400, -0.03163774, -0.10947803, 0.03036921, 0.04343887, -0.42665887, -0.01505284, 0.04941752, -0.13782160, -0.02819793, 0.04457464, -0.02195813, 0.06441513, -0.25482640, 0.06668263, -0.01672242, -0.04724250, -0.06900620, 0.11524469, -0.03171089, -0.10712888, -0.10798151, -0.06610074, -0.16643295, -0.08832936, -0.15189651, 0.01095381, -0.04530597, -0.04240334, -0.03484394, -0.06115652, -0.13995974, 0.01670393, 0.06339309, 0.05888122, 0.00535131, -0.10874606, 0.03189240, 0.02937920, -0.04765759, -0.11503935, -0.01730627, -0.03104729, 0.13876335, -0.11224630, -0.01215708, -0.08660945, 0.02211609, -0.03630905, -0.02602152, -0.00087243, -0.06802320, -0.01693748, -0.02123907, -0.05423267, 0.03727177, 0.06027410, -0.16835216, -0.04415832, 0.00199872, -0.04528118, -0.07987881, -0.16990939, -0.06416252, -0.07876762, -0.09933411, -0.00661585, 0.09576551, 0.04644351, -0.19702297, -0.05255202, -0.12991011, 0.02663761, -0.00224184, 0.16008081, -0.09512313, 0.06571338, 0.02132230, -0.09071608, 0.03210982, -0.03912129, -0.11938399, 0.11514934, 0.07951898, 0.01914841, -0.01754341, 0.00373109, 0.08974253, -0.15729308, -0.07593413, -0.03879921, 0.03402567, 0.09995465, -0.14384247, 0.09429213, 0.08961061, -0.33509505, -0.04286585, -0.01431520, 0.06557251, -0.00319814, -0.30059281, 0.00616137, -0.02882368, -0.09338118, 0.02784879, -0.10449941, 0.01142311, 0.03534007, -0.04982234, -0.03037581, -0.02564703, -0.10915610, 0.00138331, -0.02682989, 0.06256657, -0.00836788, -0.03670825, -0.01086782, -0.04101566, -0.10173774, -0.01860392, -0.02937799, -0.07045354, -0.08928045, -0.03933617, -0.13048981, -0.12135400, 0.10691800, -0.06407764, 0.01341937, -0.03056862, -0.03402815, 0.07188993, -0.03759595, -0.04093061, -0.13704902, -0.04699948, -0.02998299, 0.01180669, -0.02861975, 0.07630870, -0.06266995, -0.01991111, -0.04958840, 0.00241825, -0.02411142, -0.02207416, 0.01839490, 0.04997580, -0.01194022, 0.01762186, -0.14239217, -0.09673081, -0.04922747, -0.02527537, -0.05317595, -0.10143077, -0.15765736, -0.11926624, -0.13553582, -0.20811592, -0.05711916, 0.26497269, -0.01959366, -0.24738285, -0.08608060, -0.07992571, 0.04147243, -0.07155915, 0.07486878, -0.03708762, 0.03941049, -0.01081811, -0.05787636, -0.01429369, 0.03546583, -0.16215648, -0.07802404, 0.10798214, 0.03601297, 0.02097363, 0.03401711, 0.10888076, -0.25184694, -0.04547145, -0.12963237, 0.18404253, 0.04227228, -0.26377293, 0.11607055, 0.00380441, -0.32590446, -0.07649245, 0.01813652, 0.03449531, -0.04143630, -0.13501331, -0.05764464, -0.08949620, -0.12475986, -0.07290062, -0.02329580, 0.00650261, -0.11464446, 0.02367984, -0.11462149, -0.04486021, -0.14726667, -0.09419971, 0.02680845, -0.08005527, -0.06803048, -0.00732785, -0.06968845, -0.17455202, -0.04878071, -0.05528193, 0.10967961, -0.04945582, -0.03036611, 0.03210275, -0.09844080, -0.10580795, -0.02044102, -0.17256325, -0.07484483, 0.05429065, -0.04671837, 0.06557201, -0.13804404, -0.14178550, -0.15863736, -0.02792029, -0.15929691, 0.07839721, -0.02380192, -0.02204500, -0.08556694, -0.05500636, -0.06882428, -0.10955782, 0.01548548, -0.03456806, -0.05954112, 0.01888581, -0.03652344, 0.02035645, -0.09920915, -0.15455122, -0.05139511, -0.11549280, -0.11662094, -0.02900701, -0.09994625, -0.07608879, -0.12431379, -0.10779016, -0.01701431, 0.13390099, 0.01490799, -0.16381249, -0.17650843, -0.12301598, -0.08500336, -0.03330081, -0.14269388, 0.12745048, 0.03433391, 0.11641043, -0.10014072, 0.10791131, -0.15335688, -0.03944325, -0.00912171, 0.09229051, 0.00660807, -0.02420853, 0.04231545, 0.03643253, -0.31759381, -0.01288011, -0.10434098, 0.10223205, -0.09935689, -0.34778634, 0.08088119, 0.00068403, -0.13582380, -0.08368988, 0.08421149, -0.06187292, -0.09136407, -0.17699999, -0.02448010, -0.11022493, -0.01961948, -0.14829801, 0.19420557, -0.11188468, -0.07733033, -0.22228965, 0.02014801, -0.00772701, 0.01764814, 0.02811718, 0.04994830, -0.07313409, -0.03058243, 0.03486338, -0.08482975, -0.02180443, 0.02898108, -0.01664550, 0.15689464, 0.02814679, -0.06933827, 0.03709284, -0.04679364, 0.02513945, -0.09320946, -0.04257618, 0.06588367, 0.04909123, -0.11608680, 0.07224711, -0.00913913, -0.04972843, -0.03303510, -0.04888328, 0.01495299, 0.01484168, -0.13691257, 0.01538726, 0.04335017, -0.05024476, -0.05096074, 0.00404109, 0.10000621, 0.01557997, -0.01177004, -0.03657687, -0.14677970, -0.09906838, -0.01563630, -0.00701998, 0.06859060, -0.12432675, -0.03444802, -0.20329224, -0.00696388, -0.13778138, -0.00573097, -0.05980517, 0.04177019, 0.00942450, -0.03814129, -0.21730568, -0.02398478, -0.03645839, -0.02989299, -0.09024401, 0.06793891, 0.10503454, 0.05736964, -0.01968815, -0.04747096, 0.12038209, -0.26446536, 0.03490892, 0.01698283, 0.10687044, -0.03057252, 0.00822207, 0.10438409, 0.01915859, -0.21047090, 0.00058486, -0.32440695, -0.04836029, -0.01039466, -0.22412638, 0.02514638, -0.05709999, 0.01991903, -0.09440261, 0.16881762, -0.05071423, 0.01638761, -0.27965310, -0.05806498, 0.03543941, 0.05124879, 0.11008392, 0.07941942, 0.07326132, 0.08444053, -0.14908315, -0.03298725, 0.01547083, 0.06760948, 0.03596683, 0.03126022, 0.09385084, 0.00856508, 0.04679284, 0.02109929, 0.04604205, 0.08433206, 0.09865174, -0.01154478, 0.00288363, 0.08462481, 0.10372084, 0.05161786, 0.08972320, -0.11372098, 0.10969210, 0.12871066, 0.08133457, 0.04406797, 0.01195772, 0.07092493, 0.11619439, 0.07367414, -0.01343762, 0.14040704, -0.02845876, 0.04715643, -0.08802786, 0.02524296, 0.03491461, 0.09460983, 0.07803887, 0.04792294, 0.00798861, -0.00472995, -0.07619394, 0.01527815, 0.02534364, -0.04644683, -0.00159384, 0.12753336, 0.12026400, -0.05006513, -0.02279300, -0.06375344, -0.03128287, -0.15509850, -0.07265873, 0.06085100, 0.26269680, -0.00240985, -0.12559070, 0.04301371, -0.01069591, -0.19288673, -0.02860547, 0.19433410, 0.08329985, 0.05285405, -0.05090532, 0.02649143, 0.02116510, -0.21557543, -0.00106543, -0.26282325, 0.00146225, -0.05270185, 0.01544191, -0.00103986, 0.08757782, -0.24448797, 0.12901756, -0.40323314, -0.10208392, 0.09418896, -0.09097040, 0.08942679, 0.02744914, -0.13013442, 0.11882971, -0.05721502, 0.02881110, 0.05539004, 0.09253978, 0.02163708, 0.09817004, -0.04434669, 0.07581332, -0.16288884, 0.11187439, 0.09429719, 0.08346170, 0.07218909, 0.05116622, -0.00796536, 0.09372078, -0.16673800, 0.03196421, 0.09673539, 0.15505028, 0.09299167, 0.04103603, -0.04675182, 0.00503940, -0.02997400, -0.03260772, 0.06070518, 0.14069328, 0.05118829, 0.01189713, 0.06833318, 0.09775074, 0.02174372, -0.00489775, 0.11443845, 0.11890133, -0.09891633, -0.03241584, 0.02972164, 0.11407876, 0.00734002, -0.03466956, 0.04344161, 0.09721439, -0.00895843, 0.01611678, 0.05731116, 0.00471541, -0.01442902, 0.06049322, 0.03586277, -0.03028222, 0.01380159, 0.04060917, -0.09382640, -0.03896615, -0.07030759, 0.24511930, 0.03624266, -0.07229259, -0.01255231, -0.03071130, -0.16105556, -0.10361557, -0.03560415, 0.21469590, -0.07385723, -0.10257624, 0.00906582, -0.02318640, -0.20065233, 0.03600781, -0.05058501, 0.07191683, 0.00564852, 0.09108922, 0.04045170, 0.04459441, -0.30405945, 0.06411791, -0.17086470, 0.04605334, 0.08305013, 0.03611939, 0.11139394, -0.01314906, -0.35924739, 0.05201593, -0.13519916, 0.06956484, 0.04603055, 0.12180262, 0.03569365, 0.08644436, -0.20855367, 0.16773808, -0.21296303, 0.07005902, 0.02908264, 0.11471331, 0.01383176, 0.09192692, -0.29031295, 0.03571261, -0.09281821, 0.12429395, 0.06638019, 0.14525340, 0.02250240, 0.08828583, -0.09296452, -0.02582140, -0.00683882, 0.12232031, -0.03741282, 0.07633784, 0.05540210, -0.02121428, -0.03915288, 0.01109024, 0.09245325, -0.02205162, -0.01102706, 0.07719738, -0.04358432, 0.05676049, 0.03989619, 0.00530679, 0.06846900, -0.03320599, -0.01523175, 0.09510780, 0.01097251, 0.02487512, -0.07457347, -0.03447735, 0.00787178, 0.08100764, -0.05990485, 0.09787242, -0.01090684, 0.08291923, -0.03156145, -0.06356204, 0.01024489, 0.06199224, -0.08576173, -0.04557427, -0.04313783, -0.04251285, -0.00020141, -0.15894517, -0.21418010, 0.14146125, -0.12272879, 0.03167836, -0.11880647, -0.00163143, 0.01095605, -0.09878072, -0.17110622, -0.01121686, -0.03823392, 0.02277592, -0.03061325, -0.11433701, -0.03076426, -0.04470176, -0.07216357, -0.03382450, 0.01124826, 0.03423679, 0.01579097, 0.02856052, -0.16687959, -0.08283222, -0.08783115, -0.03789940, 0.00196335, 0.07964309, -0.03641487, 0.01141425, 0.10131083, 0.02073235, 0.01604256, 0.07111872, -0.08387755, 0.13896185, -0.01937065, 0.08131675, 0.05225142, 0.07630687, -0.05810855, 0.04585112, -0.05501570, 0.13198598, -0.01508746, -0.04691883, 0.07479718, 0.05657441, 0.05602684, 0.06834081, -0.03361139, 0.08955739, -0.05295395, 0.04983869, 0.10293876, 0.01018620, 0.07712261, 0.04308948, -0.04057996, 0.05731616, -0.03813775, -0.12039112, -0.08393817, -0.15546982, 0.03802255, -0.07159697, -0.10279910, 0.00714795, -0.15096617, -0.08739776
    .float -0.01151362, -0.03433812, 0.03445725, 0.09379627, -0.07661882, 0.06578661, -0.07804767, -0.09263173, 0.00538649, -0.08048683, 0.05028464, 0.05797576, -0.02288646, -0.03038030, -0.01321312, 0.01031010, 0.00881546, 0.11740842, 0.08867126, 0.00784176, 0.12160174, 0.06651407, 0.05779686, 0.10729779, 0.10259339, 0.19639815, 0.04761972, -0.06494464, 0.11851547, 0.12165209, 0.11359330, 0.06110809, 0.06349365, 0.16043195, 0.00795605, -0.01537508, 0.12878242, 0.00822408, 0.12051727, 0.06289555, 0.12050078, 0.02871920, 0.06955127, 0.07165454, -0.01299010, 0.02037983, -0.02817544, 0.06181401, -0.07300371, 0.04017652, -0.05130006, -0.08246970, -0.05234520, 0.04927587, -0.01505683, 0.04291747, -0.02116800, 0.02514677, -0.00683958, -0.08763956, -0.04150612, 0.00482736, -0.07015131, 0.00478285, -0.11577826, -0.13854641, -0.12803473, -0.03550021, -0.10602099, -0.16225581, 0.01044908, -0.06639393, 0.00432638, -0.21667232, -0.03508141, -0.01087376, -0.08049241, -0.04698235, -0.06945682, -0.08194998, -0.01377570, -0.03448470, -0.08361226, 0.00200534, -0.02941754, -0.00326214, -0.07162479, -0.05376083, -0.12722899, -0.08442912, -0.02728299, 0.03179393, -0.08776020, 0.01732160, -0.01498489, -0.11978505, 0.06822224, 0.04963697, 0.10443249, 0.06806692, -0.06655484, -0.06347129, -0.04607865, 0.07042741, 0.07845716, -0.05690384, -0.01342847, 0.06834067, -0.03613202, 0.07822856, -0.00369496, 0.07817448, 0.02448266, -0.02804763, 0.07485718, -0.01355060, -0.00161945, -0.01583399, -0.03130654, 0.04271742, 0.09684865, -0.00305733, 0.00111704, -0.12724568, 0.00206056, 0.15329038, 0.01114688, 0.02400308, -0.02474075, 0.02754650, -0.03106506, -0.24367318, 0.00701616, 0.12279578, -0.00317330, 0.00774531, -0.03985068, -0.02025352, -0.13376163, -0.08603073, 0.00393370, 0.05053543, 0.02752902, 0.05767601, -0.03810709, 0.04212398, -0.17889695, -0.16671060, -0.04399391, 0.04091746, 0.00327284, 0.02114891, -0.11995085, 0.06377383, -0.12248404, -0.19189870, 0.03409966, 0.02551804, 0.02437129, 0.00620788, -0.05165526, 0.08871442, -0.21007502, -0.08956306, -0.01845146, -0.03406896, -0.01857097, -0.02119105, -0.14314196, 0.02657951, -0.11248072, -0.11567053, -0.04155922, -0.03503934, -0.06903935, -0.02677963, -0.09188502, -0.13970920, -0.16004558, 0.02716412, 0.01264021, -0.11175431, 0.04708995, -0.05598416, -0.08407821, -0.29197195, -0.08114223, -0.00545587, 0.00938586, 0.03323874, -0.02192379, -0.12684746, 0.03762721, -0.01575298, 0.09814073, 0.05645164, -0.03423221, 0.06270249, -0.02872838, 0.02521797, 0.15312457, -0.06229638, 0.01247058, 0.01066349, 0.07337541, -0.02543113, 0.07410178, 0.06137700, 0.06667114, 0.02391847, -0.01223436, -0.09230067, -0.02421146, 0.01468031, -0.01781385, 0.04563320, 0.10619700, -0.02861235, 0.09485298, -0.32955235, 0.01292842, -0.00871139, 0.06932836, 0.08091418, 0.03353570, -0.02538188, 0.09893828, -0.29498327, -0.01029882, 0.03976094, -0.04392610, 0.05797525, -0.07029239, -0.03309423, -0.02968833, -0.09281737, 0.06116952, 0.02864305, -0.00299327, -0.02831035, -0.01993720, 0.02464346, -0.10317636, -0.08224170, 0.05148805, 0.08658619, 0.06995095, -0.03819665, -0.11719179, 0.03236360, -0.36767444, -0.07666254, -0.04223254, 0.12062911, 0.00452385, -0.01396664, -0.16491033, 0.07974641, -0.35413459, -0.02871335, -0.06249869, 0.16463290, -0.00974233, -0.01307363, -0.20686367, -0.00977884, -0.34632999, 0.02565533, -0.07286015, 0.11092962, -0.00828351, -0.02805316, -0.18195078, -0.05076352, -0.23360567, 0.03816566, 0.00114448, -0.12190461, 0.03554742, -0.02092934, -0.20276794, -0.17568900, -0.06305127, 0.06151380, -0.00114475, -0.19243146, 0.04318719, -0.10281909, 0.03350404, 0.06414130, 0.00091802, -0.04434915, 0.04913313, 0.04989836, -0.00630318, 0.08403360, 0.04631856, -0.04733685, 0.09351023, -0.02040863, -0.00766840, -0.01158366, 0.06237150, -0.02862402, 0.00495947, -0.06539297, 0.12990449, -0.16395219, -0.01857348, 0.02725092, 0.06535169, 0.03765154, 0.03088959, -0.03537560, 0.11124016, -0.26026806, 0.01432083, -0.01132442, 0.03467757, -0.00853931, -0.03367602, -0.01342290, 0.07442814, -0.29064199, 0.00324711, 0.03485381, -0.08112518, -0.01053789, -0.03724934, 0.03544934, -0.07015675, -0.11536835, 0.01986922, 0.02700843, -0.05886672, 0.03313870, -0.05782884, -0.01489211, -0.07282177, -0.06756930, -0.03455073, 0.15632045, -0.05857410, -0.03297507, -0.07487974, 0.03347888, -0.17542824, 0.02114593, -0.00608230, 0.14559318, 0.01044504, -0.05595254, -0.07338818, -0.01124488, -0.27351448, 0.06629290, -0.07593498, 0.06247366, -0.02692948, -0.01078267, 0.02104648, -0.03124790, -0.21833248, 0.11520788, -0.01175224, 0.09550861, 0.05366068, 0.04535898, -0.24320462, -0.02432738, -0.04034729, 0.09506402, 0.02910906, -0.11462524, 0.04265987, -0.07948148, -0.17109109, -0.04722396, -0.01967121, 0.06046079, 0.02513193, -0.24631374, 0.06993380, -0.12620120, 0.06523513, 0.01501863, -0.02529032, -0.02680470, -0.00639505, -0.01423702, 0.02952106, -0.03101204, -0.04969731, -0.02207822, 0.07651357, -0.07794310, 0.01504405, 0.06719263, -0.08444661, -0.04498573, -0.01189651, -0.02773257, 0.17682466, -0.23781213, -0.07665901, -0.06107870, 0.00278263, -0.07335833, 0.05673065, -0.05173473, 0.14794770, -0.32911423, 0.02869601, 0.09874780, -0.03974917, -0.02011569, -0.00792703, -0.08709922, 0.01942015, -0.27532786, -0.07094145, 0.03716580, -0.06014293, -0.11483906, -0.18257858, -0.10099088, 0.03194556, -0.08066089, -0.14352046, 0.02414996, -0.13988405, -0.06883438, -0.15907900, -0.12271375, -0.01333203, -0.04425431, -0.01083900, 0.16862093, -0.11255523, -0.05421169, -0.14141786, 0.03725612, -0.01598295, 0.01076784, -0.01107130, 0.18491383, -0.14482772, -0.03669673, -0.05509303, -0.01194234, -0.12778945, 0.04808246, 0.04977169, 0.20565996, 0.03144882, -0.02913842, -0.01872685, -0.01568587, -0.03460154, 0.09871102, 0.07895801, -0.03446224, 0.11531913, 0.04527627, -0.07287310, -0.02282137, 0.14638419, 0.12852430, -0.04240368, -0.10288979, -0.00899618, -0.01529714, 0.09047304, 0.05882315, 0.18677863, 0.08134712, 0.06296146, -0.15002386, -0.01162817, -0.01594731, -0.00234388, -0.12500669, 0.00151590, 0.06363595, -0.03501818, 0.03948330, -0.00337603, -0.06937130, 0.00002882, -0.06722037, 0.12200366, -0.10784511, -0.02376834, 0.07849043, -0.00104237, -0.02040432, 0.06745389, -0.06220241, 0.36851948, -0.15854190, 0.05630423, 0.06976194, -0.08637272, 0.01060699, 0.04923547, -0.11298022, 0.35441515, -0.34352025, -0.04151366, 0.08763710, -0.00463403, -0.10247980, 0.05618670, 0.01669119, 0.15903312, -0.40234426, 0.00385439, -0.02004030, -0.04745252, -0.15333009, -0.00475028, -0.14786622, 0.11785721, -0.19107318, -0.09039389, -0.06855400, -0.05243057, -0.17966583, 0.04939185, -0.06956218, 0.12778619, 0.08242095, -0.07267909, -0.09061260, -0.12253682, -0.11294997, 0.03158084, -0.19973077, 0.03904048, -0.01063586, -0.00013430, -0.05186522, -0.14785703, -0.01227612, 0.07467418, -0.08617243, 0.14194945, 0.06617337, -0.02889789, -0.08193889, -0.00830891, -0.02425305, -0.09560949, 0.00219014, 0.10703203, 0.09922467, 0.07913505, -0.23737831, 0.07886302, -0.01340814, 0.04221469, 0.03258659, 0.21849725, 0.06143444, 0.03309219, -0.21663041, 0.05181406, 0.00055446, 0.19379598, 0.15890160, 0.31261927, -0.07363632, -0.01037600, -0.02560878, -0.05816510, 0.14131348, -0.08605012, 0.00867550, 0.09420756, 0.08535466, -0.08404229, -0.07267879, 0.06332081, 0.01613702, 0.13108370, -0.04615486, 0.24892814, 0.00574472, 0.05147403, -0.10309494, 0.00657527, -0.02710174, 0.16199477, -0.14007324, 0.27804613, -0.27157590, 0.04450650, -0.11667390, -0.05826654, -0.05586996, 0.10195543, -0.10437483, 0.18819445, -0.42450854, -0.04157472, -0.08945855, -0.01556003, -0.10576544, 0.11730903, -0.01998291, 0.15431368, -0.41675755, 0.07553858, -0.12112319, -0.10367144, -0.08665650, 0.07906675, -0.03461182, 0.12963551, -0.15711656, -0.00739458, -0.18442225, -0.06824257, -0.04870109, 0.08513460, -0.00231961, 0.08528458, 0.09998196, 0.03052694, -0.20675819, -0.04207917, -0.03654088, 0.12767351, -0.11792089, 0.10178927, 0.03852270, -0.03596777, -0.21772183, 0.01984993, 0.03336201, 0.13417251, -0.11114232, 0.22546586, -0.00615425, 0.02327536, -0.26934102, 0.10083898, 0.00728480, 0.05273447, 0.02411901, 0.14097676, -0.01554323, -0.03679452, -0.20389178, 0.02867834, -0.07670332, 0.14623378, -0.03248890, 0.17825553, -0.11839935, 0.09845875, -0.15402067, 0.01008024, 0.00266989, 0.16160217, 0.11331213, 0.29049119, -0.15716144, 0.10794490, 0.19111153, -0.05107509, 0.06953697, -0.00093729, 0.08271734, 0.13090390, 0.06235547, 0.06974740, 0.03491372, 0.05435760, 0.08005828, -0.01233355, -0.02088420, -0.00745825, 0.00668772, 0.05751657, 0.00585702, 0.10565530, 0.09084707, 0.04120446, 0.03045955, 0.01016658, -0.08638018, -0.05131314, -0.17301644, 0.01664339, 0.03725960, 0.18471454, -0.01538469, 0.18882622, -0.17040560, 0.01643483, -0.23547126, 0.07315986, 0.05670652, 0.19416521, -0.04552897, 0.14848918, -0.10389428, 0.06951479, -0.21031822, -0.00776089, 0.01125092, 0.20646991, 0.00076959, 0.11810423, 0.03878453, 0.06256163, -0.09560658, 0.11597583, -0.00337482, 0.12780710, 0.07403277, -0.02250651, 0.10007046, 0.09904327, -0.07944262, 0.06393960, 0.02056054, 0.12110755, 0.08569732, 0.11936188, 0.02740359, 0.00342300, -0.09883285, 0.03737047, 0.03223613, 0.03417212, -0.03604615, 0.11819496, -0.08286573, 0.02243684, -0.12963970, -0.01078538, -0.07929061, 0.08982969, 0.10226494, 0.08580916, -0.05635475, -0.04933061, -0.08079281, 0.02289470, -0.05747124, 0.01103553, -0.01279140, 0.22382575, -0.16319081, -0.05048093, 0.04795368, -0.01540849, 0.02364041, 0.16522427, 0.15635705, 0.18731298, -0.18404873, 0.06920125, 0.30151826, -0.01154960, 0.17734119, 0.07212934, 0.02388702, 0.09723040, 0.01166881, 0.07148839, 0.05051054, 0.01536070, 0.08523123, 0.06344551, -0.01560074, -0.01211303, 0.06687962, 0.00537461, -0.06602901, -0.05381998, 0.02851173, -0.08027952, 0.00487344, 0.02111540, 0.06782958, -0.05828485, -0.08251429, 0.07507664, -0.02710271, 0.01777744, -0.07813890, 0.04791499, 0.04562813, 0.07677694, -0.11893718, 0.00795703, 0.06917419, 0.15099478, -0.07155348, 0.10643411, 0.13606393, 0.12050433, -0.06229553, 0.05384824, 0.06686450, 0.21758460, 0.05765596, 0.12056097, -0.04956500, 0.07128307, -0.14586048, 0.09675349, 0.13093071, 0.06350361, -0.00411383, 0.12283245, -0.05662207, 0.05187740, -0.12640637, 0.16008152, 0.11792906, 0.10431200, 0.09216186, 0.15862191, -0.06840250, 0.03795595, -0.07201853, 0.01434140, -0.02774247, -0.01012637, 0.06171496, 0.09948234, -0.16695374, 0.02064094, 0.06040374, 0.07677585, -0.03638548, -0.02402704, -0.00376674, 0.16419938, -0.20148683, -0.00102223, 0.18062291, -0.08008802, -0.03996267, 0.02539063, 0.08841320, 0.17775777, -0.14418980, -0.08926408, 0.24835137, -0.09758785, -0.02955821, 0.23929724, -0.04707327, 0.14129716, -0.16772096, 0.02497246, 0.16561201, -0.14997740, 0.09014031, 0.03165302, 0.07115348, 0.02818966, 0.02468913, 0.07133142, -0.03393050, 0.06696656, -0.01152828, 0.01605171, -0.01872383, 0.00032814, -0.00014738, -0.00621382, -0.11634986, -0.03208481, -0.03649826, -0.01444700, -0.03583330, -0.00485175, 0.08959267, 0.00939535, -0.10116697, -0.03110923, 0.05178682, 0.00705591, -0.07307520, -0.07359902, 0.07079309, -0.00608687, -0.02831942, 0.05024537, 0.02480231, 0.12094730, -0.02357048, 0.02381701, -0.01904438, 0.06127184, -0.10485628, -0.01824526, 0.02397964, 0.05606643, -0.02755561, 0.03224963, -0.18647209, 0.00110010, -0.06192430, 0.05007902, 0.06067490, 0.07840168, 0.02800651, 0.01275819, -0.22903800, 0.03802440, 0.04641071, -0.02171888, 0.10546009, 0.01033090, 0.13687521, 0.06911375, -0.17438377, -0.04933313, 0.13415436, -0.02382413, 0.02343791, -0.01008166, 0.08257977, 0.01959090, -0.12442531, 0.05839374, 0.24750483, 0.05924641, -0.02352569, 0.11234558, 0.15884823, 0.14566369, -0.01455488, 0.05038268, 0.30133274, -0.01120912, -0.06021690, 0.06750476, 0.01294472, 0.17887582, -0.09388673, 0.00120520, 0.26508495, -0.03479623, -0.07981119, 0.24832998, -0.05886022, -0.01274613, -0.07617290, 0.02518430, 0.08850464, -0.01681743, -0.00486730, 0.01765228, -0.10446695, -0.12923920, -0.05068139, 0.05509347, 0.00167862, -0.10527138, -0.05007123, -0.04791198, -0.01907551, -0.16863570, 0.11849102, -0.02017561, 0.05574856, 0.04616034, 0.05825794, 0.05196681, -0.01185207, -0.15795699, 0.06873739, 0.00138871, -0.07441557, 0.06094429, 0.02320200, 0.03959471, 0.07297557, -0.14286128, -0.03828053, -0.02053385, -0.02355198, -0.01842175, 0.05634274, 0.10295482, 0.06711296, -0.16848716, -0.24807529, 0.01197831, 0.12088129, 0.02768251, 0.11670569, 0.07876396, 0.05583912, -0.14791682, -0.25723347, -0.03507972, 0.04641877, 0.03623501, 0.01339004, 0.00013452, -0.01223255, -0.13015795, -0.30985388, 0.00889492, -0.02880716, 0.01093973, 0.00936840, -0.07227451, 0.13798076, -0.11994363, -0.21537711, 0.03529210, 0.09291087, -0.03552646, -0.04773656, 0.01362374, 0.18502401, 0.09075022, -0.07347643, 0.04925352, 0.22593261, -0.05097556, 0.07141677, 0.02179199, 0.18885705, 0.01108710, 0.00892076, 0.04152549, 0.09337090, 0.00166522, -0.00908936, 0.10496685, 0.04080974, 0.04931067, -0.11550130, -0.06988362, 0.16789301, 0.03208402, -0.05755119, 0.15778853, -0.03898013, 0.08909899, -0.09164417, -0.06041538, 0.09678511, -0.11578918, -0.01246891, -0.07797451, -0.21577102, 0.03118631, 0.06953862, -0.00189820, 0.00182643, -0.03686906, -0.04459475, -0.09086709, -0.23000905, -0.05855068, 0.05163164, -0.06692438, 0.00458589, -0.01827733, 0.00491917, -0.00876875, -0.15892854, -0.08894572, -0.03915340, -0.02941532, 0.07316779, -0.00369594, 0.03706103, -0.05756988, -0.10742005, -0.21955174, -0.10032091, -0.02899971, 0.08067424, -0.05113139, 0.02814083, -0.06067245, -0.04823489, -0.14107105, -0.17144676, -0.05528734, 0.03180570, -0.02985580, -0.04706470, -0.03217758, 0.09248525, -0.07170673, -0.19914903, 0.05686612, 0.05316280, -0.04147434, -0.01784779, -0.19566128, 0.08162328, -0.10067106, -0.23043664, 0.07575470, 0.01010575, 0.07407224, 0.00421763, -0.24417888, 0.11085594, -0.06842966, -0.06786389, 0.07777432, -0.01491096, 0.00163430, 0.06454033, -0.13589889, 0.03033491, -0.13378756, -0.02007728, 0.01230429, 0.16351873, 0.06003516, 0.02486307, -0.08533284, 0.05246282, -0.02361199, -0.04090619, -0.04336596, 0.12042104, 0.00418077, 0.02321588, 0.02601160, 0.06181395, 0.07249112, -0.04601147, 0.05789946, 0.13146627, -0.03170372, 0.01556214, 0.00283951, 0.08860940, 0.07052626, 0.06418594, 0.08411868, 0.07404578, -0.03537665, 0.04452280
    .float 0.03729475, 0.11534624, 0.09065277, 0.04387054, 0.12186608, -0.00594862, 0.08559746, 0.08835278, 0.11898565, 0.12445130, 0.04306158, 0.07022585, 0.03143160, 0.04837716, 0.05241746, 0.14569381, 0.09042954, -0.06715168, 0.01902485, 0.03230568, 0.07645365, 0.03004823, 0.05192613, 0.09842204, 0.09988926, -0.01216685, 0.03636889, 0.03624255, 0.07177436, -0.09289292, 0.09571574, 0.10130163, -0.02682072, 0.01536661, -0.03177361, -0.11235270, 0.07115827, 0.01593650, 0.04727833, 0.11532795, -0.02588823, 0.03610470, -0.04463482, -0.10846111, 0.02406281, -0.04507493, 0.01468776, 0.09592789, 0.01295658, 0.15770932, -0.04374633, -0.13856563, 0.01332130, -0.08754090, -0.01152540, -0.04666430, -0.07495376, 0.00856342, -0.12726803, -0.11283474, -0.04772660, -0.07445782, 0.05906313, 0.03596683, -0.10642023, -0.04771368, -0.04127482, -0.15290919, -0.03240176, -0.12909001, 0.04186323, -0.08413179, -0.09968959, -0.10890622, -0.16243769, 0.02322844, -0.09147861, -0.16228618, 0.00244220, -0.05988677, -0.08450741, -0.15773031, -0.04089832, -0.07364952, -0.12851553, -0.10916623, -0.09461443, -0.07318429, -0.03555223, -0.13733570, -0.10175650, -0.10474411, 0.02366101, -0.00330928, 0.01209290, -0.02448704, 0.10154896, 0.10234359, 0.03581459, -0.02934539, 0.04621969, 0.06152453, 0.11731321, 0.04439593, 0.06749498, -0.03009277, 0.05409730, -0.04422697, 0.01197460, 0.02088784, -0.01193103, 0.09082973, 0.07940719, 0.03374597, -0.04119045, -0.04520231, 0.03039328, 0.01537696, 0.07662258, 0.02783500, 0.04473861, -0.02193317, -0.01560562, -0.04482220, 0.00638650, 0.12231727, 0.00574756, 0.05904403, 0.03718563, -0.00224352, -0.02449456, -0.03075506, 0.01560691, 0.20471767, 0.08567638, 0.10749099, -0.01362414, 0.05446113, -0.06991362, 0.05655463, -0.02174403, 0.15237539, 0.00194173, 0.07143000, -0.05084143, 0.03975303, -0.04482661, -0.10133149, 0.07422315, 0.20338517, 0.00610346, -0.00099567, -0.10638966, 0.00364268, -0.08277179, -0.08226507, -0.00966097, 0.21755917, 0.04030724, -0.02522565, -0.03008512, -0.01686295, -0.19184366, -0.07076456, -0.01626601, 0.11161176, 0.08594830, 0.01326913, -0.04515247, -0.05481357, -0.07577937, 0.02143126, 0.03589225, -0.07754341, 0.01104484, -0.03570145, -0.09416048, -0.11733317, -0.10468049, 0.00598488, -0.03562735, -0.05368122, -0.06263059, -0.10096018, -0.10551704, -0.22412285, -0.03705711, 0.00762402, 0.04986022, -0.05889495, 0.04946505, -0.05973493, 0.19128707, 0.12709153, 0.04714549, 0.03340105, 0.09400751, -0.00543544, 0.05720745, 0.09318016, 0.06460252, 0.02173526, 0.04568545, -0.07762381, 0.03603061, 0.06068366, 0.08202044, 0.13233697, -0.02590165, 0.06856713, 0.08711527, -0.20840913, 0.05898151, -0.00355485, 0.00830869, 0.08796018, -0.00553805, 0.09342966, 0.02661545, -0.25671181, 0.09751582, 0.13332275, 0.09372637, 0.11528607, -0.02215881, 0.06512007, 0.09537710, -0.42303097, 0.08833919, 0.13634408, 0.03062085, 0.03446479, 0.00907231, 0.01399849, -0.03049749, -0.22616014, 0.06161757, -0.00104519, 0.04271097, 0.04331235, 0.05600452, 0.06909107, -0.14016075, -0.03247870, -0.01017011, 0.08488886, -0.00529867, 0.07892594, -0.09259698, 0.04427401, -0.16710623, 0.09468497, 0.06349915, -0.05632502, -0.04072428, 0.01653885, -0.12386255, 0.02556504, -0.26016465, 0.10426643, 0.01920427, -0.07760195, 0.04770742, 0.06122459, -0.06838488, -0.01651009, -0.25455153, 0.09984984, 0.04248600, -0.11877263, 0.10470583, -0.02152321, -0.01660323, -0.12400756, -0.24371587, 0.04641808, 0.02006953, -0.25288689, 0.11072462, -0.03640262, -0.10860091, -0.14975677, -0.07297155, 0.06470668, 0.05445863, -0.21965750, 0.05884970, -0.05783071, 0.08689777, 0.00553673, 0.06491883, -0.02066776, 0.08381747, 0.04301991, 0.05413403, 0.09051801, 0.10465817, -0.03064988, 0.06951194, -0.10028575, 0.04536496, 0.09078231, 0.01836943, 0.08276492, 0.08592609, 0.04134121, 0.11237094, -0.27987483, 0.12093296, 0.07159111, -0.06785914, 0.06212655, 0.13394247, 0.06453954, 0.09998959, -0.48990178, 0.05266737, 0.02958421, 0.01220445, -0.02521376, 0.11818503, -0.00266920, 0.18703736, -0.50319749, 0.04763027, 0.11944795, -0.02222491, -0.07288104, 0.09100725, 0.04085600, 0.15789500, -0.38733801, -0.05092550, 0.12269427, -0.06002676, -0.04240157, 0.13748318, -0.03057850, 0.03942685, -0.03397243, 0.10555808, -0.00851297, 0.03418990, 0.04217283, 0.18018499, -0.01695492, -0.13522357, 0.03198440, 0.05416794, -0.10875501, 0.05448033, 0.06198048, 0.14002579, -0.03061940, -0.23015276, 0.06700514, 0.04146153, -0.17575540, 0.07518457, 0.04885781, 0.07327067, 0.05030624, -0.18426624, 0.10314677, -0.00103535, -0.28914824, 0.00248960, 0.12935980, -0.07845931, -0.00110470, -0.23981804, 0.05572409, 0.07661815, -0.34174347, 0.08823825, 0.08122890, -0.28725722, 0.00489757, -0.06167927, 0.10060817, -0.01684800, -0.16437574, 0.08446624, -0.07820319, 0.07494932, 0.01363802, -0.04700302, -0.10679318, -0.00243857, -0.01593083, -0.06253914, 0.01060287, 0.14496562, -0.01037849, 0.02515535, -0.07564234, -0.03371858, -0.00352837, -0.01909264, -0.12551123, 0.08748907, -0.08589566, 0.16944003, -0.21107671, -0.06839247, 0.01204866, -0.04936138, -0.04040464, 0.15917392, -0.09542932, 0.21610786, -0.29614741, 0.00587544, 0.07529031, -0.07663242, -0.13239990, 0.05122900, -0.04494915, 0.30821788, -0.38928339, 0.07614937, -0.00238547, -0.02367494, -0.03266613, 0.15787980, -0.06335231, 0.21306430, -0.23884764, 0.01551063, -0.06538609, -0.13219006, -0.02336904, 0.18522823, -0.05708120, 0.21913217, 0.00486003, 0.02695257, -0.05934402, 0.01832915, 0.07836404, 0.18364622, -0.01389702, 0.04093547, 0.02721439, 0.06150109, -0.14499272, 0.07255501, 0.06528743, 0.25079972, 0.06018026, -0.00273121, -0.04266651, 0.04279469, -0.17102189, 0.05923638, -0.00597459, 0.11321455, 0.01947152, -0.03181433, 0.01239696, 0.00656459, -0.19952695, 0.07910625, 0.06081140, -0.15711018, 0.11354851, -0.09667930, 0.07318649, -0.00314747, -0.07724297, 0.08543681, -0.00238362, -0.23250854, 0.14533229, -0.03111688, -0.04726014, -0.05257717, -0.02920095, -0.03853247, -0.09209373, 0.09608611, -0.02977307, 0.00597287, -0.00813764, -0.12915671, 0.05775066, 0.01184802, -0.01817820, -0.05238255, -0.18540280, 0.05732960, -0.04548302, -0.09118037, -0.02987889, -0.14869289, -0.15817451, -0.01485489, -0.08998268, 0.09243213, 0.01007084, -0.09502372, -0.01445975, -0.13644075, -0.10595063, 0.06124093, -0.14130749, 0.16587777, 0.11573546, -0.11399066, 0.00893185, -0.16270633, -0.22058491, 0.09369631, -0.12704690, 0.26085576, -0.00039467, 0.00937344, -0.09960020, -0.07689711, -0.01994767, 0.15103216, 0.01737313, 0.39554039, -0.27021134, -0.02625536, 0.03422694, -0.06324005, 0.06168275, 0.10184056, -0.03236446, 0.25283122, -0.14263612, 0.11510659, -0.03034356, -0.02915820, 0.08431245, 0.12774476, 0.07696877, 0.07196496, -0.19949113, -0.04400803, -0.14618427, 0.02989430, 0.00534943, 0.14818844, 0.08449881, 0.09142755, -0.14522567, -0.07745238, -0.09191427, -0.03824333, -0.02303873, 0.10911316, 0.00552905, 0.02432068, -0.06893945, -0.04048098, -0.06670932, -0.08184895, -0.05221603, -0.04273265, -0.03609487, -0.05376621, -0.13747223, -0.10577582, 0.08082470, -0.07334083, -0.12387479, -0.00747663, -0.10203706, 0.02081126, -0.09835625, -0.09998481, 0.03860208, -0.07750050, -0.18197867, 0.05983179, 0.09963398, 0.03794197, 0.02006446, -0.03806951, 0.09088425, 0.03883099, 0.00093683, -0.13452582, -0.04363574, -0.08339290, -0.04871999, -0.20408750, -0.10762352, -0.15359417, -0.18850482, -0.18761225, -0.12815021, -0.03972384, 0.09710041, -0.15596358, -0.10288744, -0.13301560, -0.11488600, -0.19115040, -0.04667637, -0.02180507, 0.05036942, -0.18550135, -0.14109308, -0.08512951, -0.12107081, -0.06925669, -0.11569143, 0.05783843, -0.08439169, -0.05437217, 0.03616842, -0.09606219, -0.05271259, -0.03755066, -0.03255225, 0.18139057, -0.34356707, -0.01089982, 0.11319757, -0.04219190, 0.10219534, -0.00091086, 0.04590411, 0.01321363, -0.27827108, -0.00652464, 0.10198350, 0.08197105, -0.01895865, -0.02036331, 0.01676012, -0.09007483, -0.25128224, -0.05652374, 0.02359154, -0.04330505, -0.05155127, -0.00809796, 0.07561108, -0.04257131, -0.03642815, -0.09581839, 0.13264406, -0.07744745, -0.03154917, 0.02679628, -0.02197159, 0.01152229, -0.09984319, -0.02618029, 0.05455044, -0.00591040, -0.03862038, 0.00494712, -0.13406172, 0.04174361, -0.05548096, -0.14394300, -0.04942213, -0.13275237, -0.09020579, 0.12632331, -0.16319649, 0.02529685, -0.01593383, -0.05630991, -0.16462222, -0.14741801, -0.12137675, 0.03024442, 0.03190824, 0.00812759, 0.04372472, 0.06474227, 0.08343703, 0.02378250, 0.07797937, -0.02363200, -0.02462859, 0.05367241, 0.05896364, -0.14517541, -0.03434656, 0.04126569, -0.08675927, -0.09237108, -0.01473602, 0.01680658, -0.02458650, -0.10332366, -0.10133383, -0.03029397, -0.09079169, -0.19603132, -0.02632694, -0.08826849, -0.19899274, -0.11688091, 0.06552052, -0.03722748, -0.19013298, -0.12803406, 0.03002263, -0.02170380, -0.30426228, -0.08530250, 0.11658093, -0.06596407, -0.08166464, -0.14817712, 0.03033719, -0.01194126, -0.34089163, -0.06837696, 0.16483569, -0.09592487, -0.03761692, -0.23416510, 0.06357368, 0.03243517, -0.16856694, -0.04305808, 0.11171719, 0.02442874, -0.03520702, -0.02754246, 0.05090414, -0.03990904, -0.09716605, -0.10535660, 0.07578825, -0.00657908, -0.07247265, 0.01020920, 0.03194363, -0.08215924, -0.07006247, -0.08513580, 0.03088170, -0.04901643, 0.04146420, 0.03606506, 0.07534914, -0.17894335, 0.00321356, -0.04474623, 0.00604936, 0.02827386, -0.06705834, 0.03075261, -0.10210877, -0.18071440, -0.05013144, 0.01178860, -0.06204030, 0.01412655, 0.00772420, -0.15010035, -0.15737629, -0.13310142, 0.07405274, 0.07502250, -0.14925463, 0.04507424, -0.14087144, 0.02871449, 0.12813455, -0.01203491, -0.06007332, 0.01257850, 0.09343444, 0.14175332, 0.04100810, 0.04530596, 0.16139370, -0.00635719, 0.07357121, -0.01248946, 0.08177637, 0.14539804, 0.00051876, -0.08982311, 0.07124313, 0.03622437, -0.24531691, -0.02564720, 0.13757615, 0.10563028, 0.05129968, 0.00377775, 0.06960078, 0.11228672, -0.33580166, 0.00044723, 0.02278957, 0.00866677, -0.00412502, -0.07937396, 0.05832772, 0.12111264, -0.24692175, 0.00548804, 0.15158029, -0.03923285, 0.02653594, -0.13663577, 0.05049216, 0.10212696, -0.38050473, -0.07120597, 0.18233064, 0.02667900, -0.00935563, -0.04273104, 0.04874950, 0.03837679, -0.26498133, 0.01991280, 0.10174864, -0.05572964, -0.02366795, 0.13161299, 0.10110953, 0.00160879, -0.03307313, -0.05442486, -0.02902263, -0.11670892, -0.07191937, 0.02902065, -0.04963301, -0.11635207, 0.05042283, 0.05105891, 0.06065152, -0.07171469, 0.08048817, 0.02137419, -0.04318744, -0.07162042, 0.08188824, 0.09548308, -0.12243682, -0.00166692, 0.00704781, -0.07337906, -0.00076760, -0.15371688, 0.04935790, 0.06531114, -0.27456459, -0.01459682, 0.05050683, -0.18796706, -0.05362831, -0.07268348, 0.08134533, -0.05075532, -0.17250396, 0.08867449, -0.02979309, 0.08608986, -0.00239519, 0.05896299, -0.04682040, 0.10926078, 0.10642052, 0.02428344, 0.02973970, 0.00425757, 0.09980092, 0.01285509, -0.03289494, 0.12065952, 0.07964323, 0.04724053, 0.08051715, 0.02013411, 0.08133349, 0.00135279, -0.34283981, 0.01459900, 0.08966009, -0.02423538, 0.08571010, -0.09107761, 0.08896646, 0.07348309, -0.36069146, 0.01180250, 0.13522856, 0.03091238, -0.02000812, 0.03578358, -0.01544957, -0.02356040, -0.42485413, 0.04031500, 0.10586331, -0.04789180, -0.01068902, 0.04734908, -0.05736029, 0.06208891, -0.41593149, 0.03682909, 0.08121767, -0.02578601, -0.08854207, 0.06681862, -0.07005436, -0.01141675, -0.17495777, -0.01450972, 0.01814503, -0.08576179, 0.02597138, 0.09621163, -0.01429069, 0.11049499, -0.03376618, 0.08986989, -0.04412461, -0.02373551, -0.00664678, 0.05175451, -0.00118682, 0.04025273, -0.04191088, 0.03650279, -0.10004232, -0.00680090, 0.09577936, 0.10026649, -0.05694260, -0.04134426, 0.00227202, -0.01973310, -0.16763736, 0.06619978, -0.03033684, -0.07242492, -0.04326221, -0.03829844, -0.00827963, 0.01870091, -0.22740468, 0.01410892, 0.06738977, -0.31088465, 0.00905780, 0.04214961, 0.00319092, -0.07500158, -0.15004925, 0.04827295, -0.05979498, -0.04919958, 0.05772648, 0.10034089, -0.02541883, 0.06612120, 0.02838210, 0.11227790, 0.04495607, -0.01571602, 0.00383358, 0.20187624, -0.10865314, 0.02480149, -0.01460904, 0.09683014, 0.03593308, -0.00515793, -0.01450954, 0.16148545, -0.37585092, 0.00574605, 0.01396129, 0.01668629, 0.09064169, 0.04362257, 0.09275596, 0.21026866, -0.39644876, 0.07785530, 0.16105980, 0.10532624, 0.10939199, 0.06740990, 0.00120770, 0.22862710, -0.48225793, 0.01562085, -0.07298443, 0.04134146, 0.05869547, 0.12166467, -0.00728134, 0.21099633, -0.28545564, 0.09978186, -0.01228468, 0.01141190, -0.04361145, 0.08892570, -0.03715662, 0.20977953, -0.08034666, 0.03918276, -0.13934223, -0.00818684, 0.05701590, 0.11873069, 0.02488128, 0.28001106, 0.00302196, -0.02585888, 0.00850061, -0.05805232, 0.06166129, 0.15290433, -0.02390338, 0.15803322, -0.02829488, 0.06474657, 0.00767440, 0.04663876, 0.02319135, 0.05835881, -0.03910623, 0.11329289, -0.00712454, -0.03813448, -0.15864165, 0.06947006, 0.02874139, 0.11144079, -0.06371209, 0.06700096, 0.02514546, 0.02739302, -0.05168079, -0.04879398, -0.11455234, -0.11504579, 0.02169710, 0.03030470, -0.03074911, -0.03007908, -0.12615180, 0.02800468, -0.10637725, 0.12857002, 0.17454003, 0.04665780, -0.00294128, 0.08022188, 0.08487704, 0.09990297, 0.11274192, 0.09753878, 0.24551941, 0.05514941, -0.07150446, 0.13657013, -0.00512315, 0.04582512, 0.12203727, 0.01917973, 0.21997578, 0.20754811, -0.22141877, 0.00766291, 0.05037889, 0.15770157, 0.06736491, 0.12762964, 0.11898671, 0.21877600, -0.17128275, 0.03032405, 0.06282424, 0.07775731, 0.06496263, 0.14807677, 0.04037134, 0.31269479, -0.09713528, 0.05133554, -0.05000197, 0.03437343, 0.07699610, 0.16442202, 0.01014784, 0.38866389, -0.11613765, 0.04791155, -0.12365861, 0.02327952, 0.02015737, 0.22950043, -0.03291946, 0.31270623, -0.08130202, 0.03484926, -0.18421271, -0.06304097, 0.04371230, 0.23266986, -0.04867207, 0.24024782, -0.04464686, 0.02474656, -0.10529205, -0.03397562, -0.00526519, 0.22145630, -0.11424383, 0.05183674, 0.00436092, 0.01282592, -0.08764856, 0.00363465, -0.02579005, 0.17522722, -0.05849447, -0.02124424, -0.03294745, 0.06024746, -0.07263991, 0.00967823, -0.11181909, -0.09114743, -0.08965067, 0.01717559, 0.03492329, -0.12802005, -0.07262553, -0.10715898, -0.05957478, -0.02963110, 0.01942202, -0.06006032, 0.02621592, -0.02044947, 0.02569454, -0.02128792, -0.07173379
    .float -0.00874824, -0.03555091, -0.05796845, -0.11953424, -0.09282988, 0.01213999, -0.02663040, -0.02661830, -0.03694278, -0.05415419, -0.01300463, -0.12143674, -0.07251213, -0.05816801, -0.07313535, -0.03725196, -0.00713152, -0.09586711, -0.07809929, -0.09423898, -0.18221818, -0.12349251, -0.14605601, -0.17123441, -0.16752383, -0.19742772, -0.20104073, 0.02710817, -0.17178102, -0.06962307, -0.21158883, -0.28915474, -0.14725965, -0.18433179, -0.13787062, -0.05848496, -0.20967053, -0.03834108, -0.13988511, -0.20030701, -0.11981124, -0.12011483, -0.03502194, -0.08348132, -0.32071692, -0.02431815, -0.10809352, -0.16229573, -0.04473761, -0.01974039, 0.02252338, -0.01675937, -0.16337587, -0.01032433, -0.02668867, -0.11002958, -0.04882527, -0.11491600, -0.01970026, 0.04235622, -0.05785977, 0.02894786, -0.11159675, -0.10586584, 0.07351337, -0.01785079, -0.00372518, 0.04637637, -0.04503218, 0.10945306, -0.06329214, 0.10190362, 0.01651113, 0.00718160, 0.04251005, -0.03411453, -0.08344391, 0.10067098, -0.04176159, 0.08921457, 0.12820135, -0.00266270, 0.00226511, -0.03781906, -0.01212254, 0.10101877, 0.00908981, 0.06188235, 0.05571557, -0.10424853, -0.06739441, 0.00446353, 0.06626862, 0.00063028, -0.02801271, 0.11797434, 0.00725801, 0.06768234, -0.08867823, -0.13182157, 0.03726241, 0.00895843, -0.02846785, -0.01493894, 0.00569746, 0.04350867, -0.09604523, -0.13123617, 0.01744345, 0.00025446, -0.06995954, 0.03238173, 0.00031048, -0.01876535, -0.12898411, 0.00820868, -0.04330667, -0.10446400, -0.05175425, -0.09122661, 0.04184103, -0.03330651, -0.07087097, 0.06800938, -0.10779905, -0.17361940, -0.14026079, -0.20479655, -0.03214647, 0.04461723, -0.06843263, 0.19426596, -0.16962095, -0.12992713, -0.08005808, -0.25147244, -0.09903008, -0.03817166, -0.08448586, 0.23408578, -0.09323715, 0.01437316, -0.16707408, -0.12118699, -0.05804272, 0.05309593, -0.09622972, 0.22868615, -0.04820525, -0.04974826, -0.09902364, -0.05902599, -0.04613399, -0.07091372, 0.11730877, 0.25426447, 0.00824910, -0.03512347, 0.00063526, -0.06193409, 0.07464293, 0.00530995, 0.05243399, 0.21987860, -0.04081694, 0.08696765, 0.01152952, 0.04230264, 0.11720648, 0.02418824, 0.08701701, 0.17832883, 0.01002658, 0.09216595, 0.07027997, 0.05727874, 0.06616640, 0.09168386, 0.13089517, 0.09536695, 0.12899822, 0.11989716, 0.12816174, 0.12320551, 0.05547900, 0.09409789, 0.11833870, 0.08529826, 0.13527317, -0.10131663, 0.02308336, 0.13192630, -0.03230910, 0.03865352, -0.02244807, -0.14422794, 0.00646421, 0.01884894, 0.06087431, -0.00116439, 0.05212910, -0.06345049, -0.08003681, -0.04118900, 0.04473908, -0.10943440, 0.00180487, -0.01867159, -0.11417368, 0.06107660, -0.05742557, 0.09362965, -0.07292442, -0.04996780, -0.00462475, -0.05812101, -0.09700722, -0.08241821, -0.13909557, 0.35500062, -0.10891759, 0.01311103, -0.12493171, -0.09582230, -0.12082057, -0.06771263, -0.13508445, 0.52384406, -0.20571010, 0.10736741, -0.07821932, -0.05555328, -0.08436365, -0.19832109, -0.11800942, 0.48027116, -0.08496138, 0.07262465, -0.10548291, -0.13128732, -0.07547044, -0.20934270, 0.00001626, 0.43515730, -0.08736717, -0.06291071, -0.03262423, -0.08368835, -0.02843690, -0.18106329, 0.11751382, 0.39872718, -0.08455280, -0.02465855, -0.04557879, -0.10499236, -0.01614262, -0.23385295, 0.11917888, 0.31332478, 0.00645325, -0.00512056, 0.00316469, -0.00416471, 0.06044649, -0.08406160, 0.22998868, 0.19041972, -0.02407021, -0.05341041, 0.11440359, 0.00824637, 0.13065420, -0.03350714, 0.14478692, 0.07120281, 0.04842329, 0.03408772, 0.08234534, 0.13796577, 0.04362525, 0.10341986, 0.04436357, 0.07790654, 0.01701576, -0.06073052, 0.09955227, 0.10180380, -0.04825975, -0.02759967, -0.10231961, -0.15330279, 0.01267099, 0.02828282, -0.08567282, 0.06270479, -0.08009819, -0.00714625, -0.12223528, 0.00492144, -0.07616980, 0.02309475, 0.03718269, -0.00577443, -0.11968195, -0.07204114, -0.04238833, 0.16435479, -0.12980253, -0.07862344, -0.00701529, -0.05748763, -0.15545359, -0.07183520, -0.03879513, 0.31197107, -0.05163933, -0.05322586, -0.04226597, 0.01102710, -0.08921669, -0.13346988, -0.07659112, 0.47603595, -0.05716966, 0.00373543, -0.08187520, -0.04680399, 0.00559510, -0.18153302, -0.21688595, 0.28239894, -0.06740610, -0.19603114, -0.03874868, -0.16624525, -0.06774320, -0.10634597, -0.09280629, 0.37878275, -0.04571335, -0.29985708, -0.06937827, -0.05358765, -0.02094596, -0.07856312, 0.04409501, 0.31559640, -0.08459602, -0.08352967, -0.03071126, -0.07704949, 0.09219210, -0.19866806, 0.22690283, 0.33772230, 0.05016849, -0.13200583, -0.04967389, 0.03618729, 0.05136649, -0.16527265, 0.24004462, 0.13717301, -0.01216999, -0.25434223, 0.07754111, -0.00646666, 0.12862405, -0.12529658, -0.02181965, 0.08609103, 0.01002184, -0.19084495, 0.09349275, -0.03528436, -0.05928550, -0.00709034, -0.13821895, -0.01278047, -0.01569417, 0.00877124, -0.01265570, -0.03798641, -0.11690044, -0.06389377, -0.00571936, -0.08982237, -0.06694063, -0.00911846, -0.02588022, -0.02847142, -0.18787618, -0.03092594, -0.00752007, 0.06732046, -0.06188736, -0.01527859, 0.00961496, 0.01428256, -0.04715616, -0.00651575, -0.01764154, 0.21017382, 0.04351940, 0.01152228, -0.03720718, 0.02852607, 0.09734251, 0.01189835, 0.06497282, 0.14606811, 0.00045714, -0.04279114, -0.01255110, 0.05110055, 0.04427961, -0.02673444, -0.04734239, 0.16303359, 0.02655226, -0.11376467, -0.01671754, 0.05612132, 0.05287127, -0.10143755, -0.11357194, 0.20133919, -0.03656301, -0.31504914, -0.03421038, -0.06580672, -0.11694948, -0.10722408, -0.17951059, 0.16433120, -0.06799491, -0.17465252, -0.04497601, -0.00105481, -0.11332687, -0.15674087, -0.08942277, 0.19175859, -0.10018296, 0.09787223, 0.01399612, -0.02767251, 0.05044140, -0.08741150, -0.00337829, 0.19202513, 0.02872064, -0.13925140, 0.02011089, 0.06618647, 0.02529213, -0.17777188, 0.23266304, 0.08981670, 0.04033324, -0.31265947, -0.00401839, 0.00135807, 0.00628108, -0.12118936, 0.02436308, 0.03080309, -0.08776758, -0.26290393, -0.02173286, -0.10506121, -0.05776908, -0.10497323, -0.10460091, -0.11274274, -0.09493066, -0.06736063, -0.12550183, -0.02820676, -0.08837644, 0.01445287, 0.01947077, -0.09507193, -0.02533338, -0.06100756, 0.05964388, -0.02505595, 0.03385682, 0.02003608, 0.09501282, -0.02153372, -0.01992468, -0.05865296, 0.09404433, 0.04724257, 0.14053582, 0.10757209, 0.01628939, -0.02992607, 0.06233100, -0.02155654, 0.12310527, 0.10959917, 0.07895555, 0.07365168, -0.07089318, 0.08363481, 0.09782019, -0.08925740, 0.02092903, 0.02491040, 0.15659888, 0.08778357, -0.07197508, 0.02449359, 0.06081001, -0.02734910, 0.12280773, 0.08055275, 0.03426749, 0.06290054, -0.29801226, 0.04135508, -0.08289870, -0.30104348, 0.07034469, -0.13015333, -0.07590914, -0.05184579, -0.13152419, -0.12878625, 0.03074610, -0.15521181, 0.00315890, 0.00180197, -0.04316830, 0.00433245, -0.17392319, -0.00130061, 0.06524105, 0.00856616, -0.11385404, 0.12196504, 0.03841509, 0.00186308, -0.05711216, 0.09876824, 0.02413742, -0.07429573, 0.03200350, 0.01080956, 0.12472580, 0.03262097, 0.17270625, 0.02205843, 0.04272691, -0.23081122, -0.02197688, -0.02229988, 0.04607708, 0.01813780, -0.02033357, 0.10632548, -0.02495055, -0.13467838, -0.05180737, -0.08372476, 0.03875483, -0.07015894, -0.13471565, -0.13172485, -0.02799738, -0.06747530, -0.10268424, -0.11330259, -0.06369227, 0.00524967, -0.11444317, -0.05180254, 0.08397076, -0.08845997, 0.03188116, -0.04126159, 0.03949041, 0.02094507, 0.04871303, -0.17783390, 0.10226833, -0.05644396, 0.08837511, 0.03198943, 0.09945824, 0.09102359, 0.01242971, -0.17790566, 0.11601149, 0.03611319, 0.06833661, 0.13552321, 0.07639000, 0.01885756, -0.11264074, 0.03806701, 0.05656077, 0.06571171, 0.08583587, 0.05615852, 0.01124883, 0.02218454, -0.12992130, -0.04169207, 0.02782377, -0.01083836, 0.12464299, 0.05818748, -0.17590155, 0.08707784, -0.24245423, 0.00728354, -0.02235815, 0.04759708, 0.07159089, -0.10920610, 0.03541884, 0.13215038, -0.07858887, -0.29432231, 0.08350774, 0.05837834, -0.04107500, 0.00344506, -0.01445069, 0.02377559, -0.07298618, -0.05272290, 0.10405155, -0.05406086, -0.05614520, 0.11078864, -0.02595568, 0.05613127, -0.07159745, -0.00113879, 0.13939860, -0.13930950, 0.07349791, 0.04011424, 0.02691561, 0.12495749, 0.00879971, -0.03570154, 0.11972196, 0.09816765, -0.00238742, 0.04933240, 0.01587815, 0.03619158, -0.04554041, 0.02750828, -0.02188188, 0.05213184, 0.09544459, 0.02118154, -0.08453467, -0.01050830, -0.05709265, -0.05301613, -0.09122830, -0.00112337, 0.01905266, -0.02056052, 0.06673726, 0.02051640, -0.12537774, -0.17349337, 0.04997291, 0.00982460, -0.07550774, 0.00828604, 0.03856764, -0.02993044, -0.01265676, -0.17124596, 0.13793875, -0.00284756, -0.01081886, 0.08182679, 0.02525727, 0.07255761, 0.00496250, -0.32767230, 0.07671695, 0.05316511, -0.03002639, 0.00669294, -0.09279910, -0.01503505, 0.02781844, -0.15796843, 0.01464643, 0.03834681, 0.07318041, 0.07384750, -0.13981007, -0.00821144, -0.07203197, -0.10479352, 0.08768507, 0.03634356, 0.05086501, 0.01173657, -0.06331103, 0.01887579, 0.02892950, -0.17871886, 0.07332940, 0.04597645, 0.02089928, 0.00464798, -0.04551340, 0.01567635, 0.06397754, -0.15073313, 0.08901744, 0.04190559, 0.05361916, 0.08923095, -0.03122053, -0.02844374, -0.08004369, -0.03423752, 0.05103655, 0.04697616, 0.01480328, 0.02648898, -0.05833978, 0.12081017, -0.02924524, -0.04368078, 0.03579180, -0.03882068, -0.00409756, 0.08637709, 0.06259443, 0.07790487, 0.11098467, -0.02453085, -0.00851057, 0.08394285, 0.10455251, 0.02985543, 0.02352727, 0.12351304, 0.00526244, 0.05495254, -0.07060049, -0.04052371, -0.00187027, 0.00280013, -0.03968607, -0.04086934, -0.04337683, -0.01587724, -0.10052382, -0.06634253, -0.09542137, -0.04140687, -0.03649126, 0.01406391, -0.03486702, -0.09507867, 0.00158540, -0.04161328, -0.04585331, -0.03321470, 0.00232084, -0.09904049, 0.10273500, -0.15257461, -0.01768932, 0.04774006, -0.04618254, -0.08555542, 0.04697413, -0.02656024, -0.02795397, -0.24044150, 0.05744085, 0.11215166, 0.02042138, -0.01658959, -0.14289974, -0.12352983, 0.09223702, -0.17006135, -0.06122336, -0.07529078, 0.06347191, -0.08248336, -0.10429025, -0.03615888, 0.02085790, -0.09497555, -0.08451560, -0.05459433, -0.05253589, -0.10643224, -0.13545091, 0.03468164, -0.02324078, 0.01153685, -0.08076996, 0.08029214, -0.08064058, -0.03337182, -0.07643427, 0.01357104, -0.14914761, -0.02619920, -0.01846219, 0.07923864, 0.01045086, 0.06549802, -0.13699174, 0.01850223, -0.31454021, 0.02273152, -0.05158529, 0.01017098, 0.02761647, 0.03651695, -0.15017490, -0.00340881, -0.15766497, 0.01660548, -0.02903674, 0.02625549, -0.00047074, -0.01359491, -0.05238251, -0.04549158, -0.06694618, 0.02516405, -0.05544567, 0.04986553, -0.00949757, 0.01507494, 0.01487554, -0.07644314, -0.08323036, 0.08859811, -0.07375296, 0.10552173, 0.00304857, -0.09632453, -0.02887571, 0.00747749, -0.12128396, -0.09457003, -0.04997845, -0.04480997, -0.03427618, -0.10074453, 0.02477030, -0.03916915, -0.09601266, -0.13200895, -0.06557951, 0.00259376, -0.04871755, 0.01569862, 0.00809036, -0.12400480, -0.12474503, 0.03239989, -0.09996936, -0.04160384, -0.10414659, -0.01653807, -0.04498314, -0.09856717, 0.02629708, -0.00771490, 0.02736191, -0.11760400, -0.06046825, -0.00293323, -0.01263301, -0.15021674, 0.04553849, 0.17123687, -0.08891243, -0.15498312, 0.00981712, -0.10404279, -0.03392663, -0.15549167, -0.03066967, 0.17333125, -0.12652305, -0.09683480, -0.08448143, -0.11809601, -0.12771307, -0.08134165, 0.00513042, 0.23979977, -0.08226233, -0.02406606, -0.10932068, -0.14345673, -0.08362193, -0.10452687, -0.09968439, 0.14593555, -0.00720174, 0.07404902, -0.07596802, -0.04418419, -0.18653694, -0.04763870, -0.26151183, 0.02642483, -0.06248821, 0.02721545, -0.00251275, -0.08425905, -0.16859303, -0.07630412, -0.31582063, -0.02509279, -0.03973437, -0.01272944, -0.02155338, -0.10255236, -0.21033983, -0.08684769, -0.18669708, -0.01886533, -0.01650234, 0.08024172, -0.06410257, 0.02221219, 0.09117986, 0.00479666, -0.05596542, -0.03663245, -0.10476757, -0.05608882, -0.00223126, -0.06629325, 0.03012280, -0.18243255, -0.01950758, 0.00816385, -0.11917036, -0.10296786, -0.06584232, -0.14209150, -0.00947718, -0.10704244, -0.11286883, -0.06571915, -0.00970714, -0.00750585, -0.08256082, -0.08450933, -0.05282118, -0.00649248, -0.05907926, -0.03681412, -0.10335436, -0.09423165, -0.04168873, -0.01385186, -0.00289785, -0.13648416, 0.06942830, 0.07694408, -0.06328192, -0.02708897, -0.01557380, -0.00247313, -0.02657641, -0.05308115, 0.04100890, 0.19649918, -0.13666727, -0.14670493, -0.06625643, -0.02216588, -0.06814011, -0.11462106, -0.12371337, 0.11258542, -0.02030885, -0.19730733, -0.09236153, -0.03446228, -0.04611121, -0.17495367, -0.01437009, 0.20954302, -0.08850658, -0.12785797, -0.07040428, -0.03931153, -0.04319604, -0.03115952, -0.14586310, 0.14182620, -0.04044234, -0.04568802, -0.11512063, 0.03977520, -0.04205203, 0.06022369, -0.29570350, 0.19183175, -0.03332645, 0.01085415, 0.00117585, 0.06687661, -0.04219216, 0.00262973, -0.17197189, 0.10226554, -0.00413135, 0.04294082, -0.05194750, -0.00477448, -0.02023582, 0.14618953, -0.14423341, 0.05999569, -0.03146476, 0.02286420, 0.05658962, 0.05924616, -0.08001370, 0.08679026, -0.06250363, 0.05869789, 0.03131130, -0.03299584, -0.01011356, 0.02779524, -0.02206776, 0.00734098, -0.03814541, -0.03380541, -0.06176637, -0.05830142, -0.04938228, -0.07960949, -0.10218093, 0.02118764, -0.05171860, -0.13526711, -0.11839755, -0.07605220, -0.01970452, -0.03905994, -0.00475899, -0.05085053, -0.03533934, -0.08397806, 0.00860627, -0.07181294, -0.13646100, -0.04658985, -0.03371028, -0.09167277, -0.07413005, -0.02363899, -0.05982746, -0.12351199, -0.11648779, -0.08090816, -0.05233247, -0.07834399, -0.19247320, 0.01423461, -0.05729591, 0.02553131, -0.11414354, 0.03863357, -0.07502082, -0.11673471, -0.23625849, 0.06507155, -0.08802168, -0.00852145, -0.09870157, -0.13379101, 0.03752251, -0.03206010, -0.33475167, 0.10820406, -0.04327297, 0.04702718, -0.04264583, -0.07573436, -0.01737038, -0.01973074, -0.40144596, 0.01100246, -0.07171241, 0.04358073, -0.00102457, -0.05589256, -0.04066973, -0.02778215, -0.42030868, 0.04074611, 0.02272786, 0.05929201, -0.02412664, -0.09715912, -0.05879395, 0.00522092, -0.17262459, -0.15124553, -0.09824958, 0.12341538, -0.04128207, -0.05248371, -0.14431149, 0.02592698, -0.17349187, -0.05529618, 0.00076534, 0.05470606, -0.05315773, -0.01316028, -0.02224902, 0.01660699, -0.10337315, -0.04430284, -0.13686350, 0.10643195, -0.05585735, 0.06554084, 0.17095996, -0.09968135, -0.13054076, -0.12033597, -0.02740220, -0.05346625, -0.04159518, -0.11879718
    .float -0.09996334, -0.15950549, 0.10738368, 0.00871792, -0.02552467, 0.07725674, -0.09485914, -0.06074239, -0.13293870, -0.17423593, -0.04369379, 0.11707434, -0.21368082, 0.00364186, -0.13859230, -0.20588244, -0.04768712, -0.04972631, -0.03296150, 0.10812419, -0.22568510, -0.03569010, -0.15483485, -0.12607050, -0.13602160, -0.00977543, 0.07300683, 0.08698767, -0.11228790, -0.04210797, -0.05277259, -0.04247033, -0.01754211, -0.09855553, 0.07374012, -0.04987752, -0.11966970, -0.05046993, -0.03646021, -0.06970974, -0.10535468, 0.01136852, 0.08008875, -0.15493323, 0.05388388, -0.02796884, 0.04642053, -0.05423843, -0.01547670, -0.04592847, 0.10115182, -0.20427305, -0.03346458, 0.00509127, -0.03946678, -0.00765573, 0.14220460, -0.00528895, 0.14516732, -0.05664314, 0.11524490, 0.03353954, 0.03195133, 0.00838194, 0.03558525, 0.05679776, 0.13803919, 0.16152284, 0.16201678, -0.10947905, 0.03361782, 0.10174565, -0.01880286, 0.04512602, 0.08475710, 0.11780501, 0.08112285, 0.03060495, 0.03684337, -0.01833058, 0.06666327, 0.01461217, 0.05682209, 0.09544154, -0.00041904, 0.12303327, -0.04783965, 0.07506771, 0.08403053, -0.01018168, 0.06814548, 0.04644057, 0.07299986, -0.02190862, -0.02013821, 0.06496304, -0.20135924, -0.10541425, 0.00163591, 0.07788774, -0.10192324, -0.02000198, -0.08448901, -0.05323145, -0.19685902, -0.03674031, 0.00057054, 0.09724165, -0.06583636, -0.03408438, -0.05544429, -0.03447209, -0.12292587, 0.02110982, -0.06611626, 0.00403974, -0.15237431, -0.07073057, -0.08392273, -0.02095034, -0.04748623, -0.00847875, -0.02542921, -0.04371933, -0.08177324, -0.01577594, 0.06147468, -0.07308821, -0.08421967, 0.02985315, 0.02427386, -0.06461135, 0.04873840, 0.01883772, -0.04624748, 0.01280283, -0.03799313, 0.00352023, 0.03962074, -0.25594780, 0.00185191, -0.10321766, 0.02055138, 0.01822377, 0.05746047, -0.06264566, 0.17172368, -0.26486111, 0.07125525, -0.09487706, 0.00846389, 0.07759763, 0.02039573, 0.03615857, 0.17411941, -0.01831314, 0.09067600, -0.16151698, 0.08803660, 0.01255745, 0.06289332, -0.00675801, 0.09389623, -0.02521035, 0.04445756, -0.04559927, 0.01528778, 0.08322994, 0.03416261, 0.06113041, 0.10627352, 0.04577219, 0.09402243, 0.05627254, 0.04542556, 0.04134730, 0.08359564, 0.08263980, 0.09471315, -0.04981420, 0.06272559, 0.14788800, -0.01386494, -0.00297947, 0.00368078, 0.12507850, 0.09246635, -0.06514888, 0.02586643, 0.13670100, -0.05495558, 0.06670266, -0.26621708, -0.12968491, -0.01736616, 0.00696015, -0.11269481, -0.09750298, -0.09898101, -0.27066305, -0.16369286, -0.06889652, -0.13083377, 0.02256880, -0.10116269, -0.07817447, -0.18334597, -0.03009835, -0.16511784, -0.04103576, -0.00162543, 0.08630279, -0.06241865, -0.09888147, -0.02722764, -0.00377163, -0.09076440, -0.04871790, 0.04679535, 0.05891206, 0.06130128, 0.01709244, -0.06083696, -0.05074800, -0.07178693, 0.02920732, -0.04027706, 0.14195961, -0.02529121, 0.01444584, 0.02113032, -0.03424612, -0.11259673, -0.01982665, 0.15894915, -0.07080553, 0.03125562, -0.04160196, -0.02274184, 0.01413203, -0.07501283, -0.02056547, 0.14864722, -0.25071517, 0.08727522, -0.13403602, 0.00661221, -0.00423956, 0.09170193, 0.09982707, 0.22210649, -0.29791543, 0.01235512, -0.07491958, -0.03938631, 0.08404537, 0.11663422, 0.12559210, 0.38115627, -0.16984454, 0.10477556, 0.02829241, 0.03763433, 0.00967088, 0.13057967, 0.01228440, 0.38842243, -0.11295033, -0.02282654, 0.05845241, -0.02017989, 0.03680978, 0.20271505, 0.11493040, 0.32939684, -0.16323006, -0.00423385, 0.09268058, -0.09028884, -0.04028540, 0.13617595, 0.13693228, 0.21540697, -0.18301362, 0.02382921, 0.22305939, 0.00650894, 0.00483098, -0.29620388, -0.24910255, 0.04571568, 0.09411857, -0.15113175, -0.11545436, -0.12616304, -0.20804651, -0.10599691, -0.12175787, -0.02188780, 0.12093549, -0.03035240, -0.03665438, -0.05061187, -0.09780511, -0.13161543, -0.02551867, -0.15322398, 0.14010295, -0.09323343, 0.00399224, 0.04157347, 0.05661755, -0.10386987, 0.02475015, -0.14159377, 0.05482420, 0.02508916, -0.14348717, 0.02841144, 0.05865974, -0.07488321, 0.08833547, -0.05204430, 0.06812145, 0.09558003, 0.04658175, -0.02159556, 0.09392425, -0.09617941, -0.01761143, -0.01012253, 0.01416165, 0.08509003, 0.06603683, 0.04955163, 0.07892518, -0.19669123, -0.04161805, 0.14267202, -0.18292840, 0.08189791, 0.03555176, 0.00618616, 0.01349428, -0.28495353, 0.03070952, 0.11177518, -0.38480961, -0.06324796, 0.07940111, -0.04229574, -0.02585283, -0.19717738, 0.13444388, 0.38168165, -0.37108755, -0.05291262, 0.15860166, -0.05971157, -0.00093076, -0.05318424, 0.03582248, 0.28714198, -0.32487845, -0.07567119, 0.15826984, -0.15639415, -0.06930725, 0.09184807, 0.11021557, 0.37537691, -0.36635128, -0.05051400, 0.08818644, -0.01227335, -0.00085356, 0.24324560, 0.08507406, 0.31373590, -0.16863869, 0.00650378, 0.10753979, -0.00676718, 0.15197730, -0.25178167, -0.03914581, -0.03133093, 0.08809824, -0.12714289, -0.16700312, -0.11520982, -0.10101426, -0.12769483, -0.04337359, -0.08986139, 0.11438414, -0.00213222, -0.10674711, 0.02115342, 0.02250126, -0.15547222, 0.07563045, -0.11309242, 0.14756761, -0.07505313, -0.03842948, -0.02281016, 0.07925089, -0.06753773, 0.06456310, -0.18754447, 0.07878591, 0.00950854, -0.05821533, 0.09306821, 0.08274496, -0.12013821, 0.11496689, -0.08477563, 0.10793161, -0.04245670, 0.10956600, 0.03548517, 0.01486187, -0.04397013, 0.03054945, -0.00774604, 0.05555392, 0.07057945, 0.12317678, 0.01047722, -0.03430852, -0.19391784, -0.01156962, 0.06171700, -0.10630447, -0.02805560, 0.09907230, -0.01273682, -0.03616199, -0.30810374, 0.13869719, 0.03283542, -0.36412969, -0.10351445, 0.21986675, -0.11709167, -0.13884589, -0.34537575, 0.02177219, 0.05065023, -0.44307062, -0.22831027, 0.23639992, -0.18346114, -0.15991762, -0.15551805, -0.10379981, 0.01390443, -0.38327363, -0.17009415, 0.23425083, -0.16360170, -0.17065190, -0.02664570, -0.01240774, 0.20024528, -0.44866449, -0.08751832, 0.14588465, -0.22307190, -0.10649801, 0.28054258, 0.02462500, 0.21190231, -0.22818145, -0.07142595, 0.25468951, -0.14341156, 0.05834616, 0.02434989, -0.05446652, -0.02589216, 0.03597828, -0.05680613, 0.02639244, -0.02900498, -0.07970702, 0.03348586, 0.06492051, -0.02787586, -0.01256876, -0.00365976, 0.01567344, -0.07255868, -0.05447533, -0.06857132, -0.01368186, -0.14286439, -0.08166064, 0.05614617, -0.02394431, -0.01714657, -0.01727344, -0.12760597, 0.04165883, -0.19473644, -0.15662402, 0.06518114, 0.03036425, 0.09382176, 0.06269024, -0.00320127, 0.09295306, -0.04047010, -0.04244730, -0.00183668, 0.12373679, 0.05207139, 0.01425815, -0.02611220, 0.11799141, -0.06390434, -0.00677817, 0.06685098, 0.10150719, 0.11230569, 0.01436098, -0.20455168, 0.09910905, -0.04021849, -0.08462317, -0.03857505, 0.20611686, 0.10478017, -0.02466647, -0.21871291, 0.14409941, -0.06038973, -0.20710520, -0.13882798, 0.21144442, -0.04234340, -0.14573486, -0.22482099, 0.03896618, -0.16655491, -0.18683764, -0.09391931, 0.35685396, -0.12179122, -0.06269688, -0.27628630, -0.00962259, -0.23237787, -0.23998848, -0.21362022, 0.16778821, -0.13072509, -0.17766352, -0.13356893, -0.07792474, -0.01084841, -0.22999537, -0.05622630, 0.05225345, -0.07076572, -0.16400649, 0.10197481, -0.11524030, 0.07694684, -0.15737551, -0.04694062, 0.17162125, -0.10997907, -0.02618606, -0.08070285, -0.09045144, 0.08860257, 0.06486222, -0.06145823, -0.04391393, -0.00164261, 0.00979862, 0.02642982, 0.01910494, 0.05801413, 0.01766690, 0.00924738, 0.04232022, 0.00382131, -0.08019213, -0.02261287, 0.07625667, -0.04011571, -0.12944789, -0.02803060, 0.08204602, -0.06231698, 0.02039093, -0.11223556, 0.08723114, -0.18660828, -0.29972962, 0.07197144, 0.15573704, 0.06273320, -0.00274530, -0.06950837, 0.06412225, -0.23123971, -0.11541858, 0.00479969, 0.12959619, 0.02049053, 0.00741145, -0.18924083, 0.03908264, -0.09483840, -0.14644316, -0.00815586, 0.07439326, 0.01967420, 0.02959423, -0.21132912, 0.02151625, -0.10087643, 0.07249716, -0.12427454, 0.13730077, -0.04774372, -0.10476157, -0.20002715, 0.09257068, -0.19292004, -0.04179617, -0.08298136, 0.21794058, -0.06428471, -0.06919162, -0.12201155, -0.01233591, -0.34420183, -0.04459435, -0.05870063, 0.19287337, -0.03046320, -0.06414140, -0.15416293, -0.06433143, -0.28034627, 0.00240741, -0.04165689, 0.05030539, 0.02644555, -0.06285277, -0.13496876, 0.01107512, -0.06366519, -0.00755458, -0.00549870, 0.03833438, -0.05587784, -0.10117786, -0.10547419, -0.11468658, 0.08070655, -0.06750274, -0.01703486, -0.09413204, -0.08459035, -0.05809097, -0.07746191, -0.02326845, 0.14222090, 0.10736150, -0.06716484, -0.02350340, 0.03707821, 0.01445091, 0.01644661, 0.15285617, 0.11995893, -0.01779992, -0.04128368, 0.07325865, 0.09443811, 0.05460362, -0.02878897, 0.05542062, 0.13833590, -0.19605902, -0.05834590, 0.06563487, -0.03538847, -0.07017971, -0.06078745, 0.08438154, 0.00566331, -0.24914546, -0.05983904, 0.23161513, -0.04803776, 0.06702428, -0.07920375, -0.02194558, -0.04161554, -0.27495351, 0.00578183, 0.19836111, -0.01154443, 0.05194144, -0.12061786, 0.05051027, 0.03808586, -0.02390123, -0.12438647, 0.02547066, -0.11894076, 0.00433996, -0.05032306, 0.03656241, 0.06704491, 0.00035723, -0.06998519, 0.10131060, -0.05525630, 0.00763567, -0.05981041, 0.02235993, -0.00217887, 0.04680779, -0.04545145, 0.07329269, -0.08503092, 0.00166266, -0.08726810, 0.00628097, -0.10378134, 0.08149464, 0.02302754, 0.07516026, 0.02846040, -0.05328861, -0.17454858, 0.05553761, -0.10612296, 0.00232789, 0.03571998, 0.00320835, 0.00292667, 0.06005520, -0.15327086, 0.00096416, -0.07625340, 0.12230130, -0.01842317, 0.04783191, -0.01622804, 0.08532252, -0.09701634, -0.00562152, -0.02630711, 0.07010955, 0.02623777, 0.01683958, -0.03562707, -0.03809417, 0.14321685, 0.05968920, 0.09189092, 0.06830072, 0.02442749, 0.05670993, -0.04698423, -0.03708862, 0.10025647, 0.11280017, 0.11591326, 0.00344537, 0.04075965, 0.07008824, 0.03424640, 0.01902788, -0.01943235, 0.08874255, 0.15399912, -0.11186001, 0.00685410, 0.08151965, 0.08784404, 0.02294212, 0.08290448, 0.06593476, 0.26516530, -0.24997632, 0.01196235, 0.20868722, -0.04466398, -0.00402906, -0.01835580, 0.12155100, 0.11725644, -0.22090515, -0.08395039, 0.06275400, 0.01662953, -0.10802034, 0.03040768, -0.05356045, 0.06818503, -0.08161180, 0.03974584, 0.10020884, -0.11218350, -0.02463205, 0.09944405, -0.08196442, 0.10294858, -0.00713213, -0.01696260, 0.03808195, -0.04141790, 0.00105495, 0.02737927, 0.03099382, 0.14750251, 0.07071241, 0.01271815, 0.08100701, -0.01991155, 0.01760808, -0.04487325, -0.06708390, 0.04531270, 0.05912574, 0.03667667, -0.03519351, -0.02036950, 0.07658902, 0.01343951, 0.03373028, 0.02790898, 0.12113706, 0.05275911, 0.00472387, 0.09653345, -0.02946829, -0.11440096, 0.00973101, 0.05727591, -0.00372329, -0.00395053, 0.00477689, 0.10801028, 0.03744755, -0.14123529, 0.01343025, 0.00286359, 0.02078618, 0.02640835, -0.10742968, 0.05221231, 0.02990154, 0.01342692, -0.05761805, 0.10362037, 0.13646075, -0.02691096, 0.08489270, -0.03908234, 0.01680953, 0.04169681, 0.04641216, 0.08775650, -0.01503384, 0.02165713, 0.07961953, -0.06639742, -0.02106255, -0.00447958, 0.12405799, 0.21067537, -0.02393221, 0.08054429, 0.12533101, 0.03052943, 0.02001836, -0.08467418, 0.00806525, 0.26660097, -0.23905122, 0.02333147, 0.11990667, 0.09299223, -0.02960572, -0.03581423, 0.02551666, 0.25588509, -0.22328110, 0.08463773, 0.18244553, -0.01829445, -0.07243858, 0.02361432, 0.01057759, 0.19579490, -0.13534680, 0.09308909, 0.09667697, 0.00607945, -0.04589505, 0.01247167, 0.03418751, 0.12978727, -0.04231013, 0.00664967, -0.15865791, 0.03477260, 0.01027583, 0.11027089, -0.00838477, 0.08162789, 0.03762947, 0.01276533, -0.12222876, -0.03749976, 0.09898804, 0.01701453, -0.05033746, 0.03368049, 0.08241905, 0.10189506, -0.14306585, 0.04026584, 0.00595911, -0.00666318, 0.03135234, 0.04845400, -0.04335104, 0.04175226, -0.06171171, 0.10137080, 0.00538375, 0.01759630, 0.08149437, 0.10386187, 0.01897917, 0.04595914, -0.03681704, 0.04797999, 0.10199566, -0.01494542, 0.13213590, 0.07993372, 0.08588743, 0.12642545, -0.04003263, 0.11382002, 0.04130075, -0.08557484, -0.05022176, 0.07905826, 0.01475256, -0.08070876, -0.04257827, -0.02674932, -0.01708536, 0.01507512, -0.04256959, 0.06633743, 0.10772010, 0.02128934, -0.04228714, 0.00486967, -0.08192850, -0.04866385, 0.02560246, 0.16652901, 0.01239737, -0.03868494, -0.01681843, -0.02617114, -0.00218469, -0.02694173, 0.00739051, 0.24769431, -0.13579360, 0.09437313, 0.08386586, 0.01403758, 0.05342797, -0.00173945, 0.02276267, 0.24078237, -0.13981089, 0.04258182, 0.08059577, 0.09010833, -0.00022760, 0.00010817, -0.03186079, 0.24415028, -0.17138650, -0.00708459, -0.07157342, -0.00951060, -0.03131609, 0.07828337, -0.05037063, 0.20216608, -0.00792419, 0.01674892, -0.08464109, -0.00317479, -0.01113140, 0.01189653, 0.02784963, 0.06344324, -0.00375623, 0.10141163, -0.10069741, 0.06097180, 0.02210424, 0.06403690, -0.04738471, 0.07357410, -0.01169535, 0.01719064, -0.17582875, -0.02778922, -0.01561840, 0.06481659, -0.03473690, 0.24628247, 0.01280651, 0.00072030, -0.05850372, 0.02770562, 0.00948928, 0.14152731, -0.02945350, 0.11876714, -0.01553210, -0.03090164, 0.01741759, 0.02548676, 0.00803958, 0.12674341, 0.08770218, 0.10092903, 0.16653043, 0.04307648, 0.05297402, 0.14882194, 0.03245310, 0.08079511, -0.07642838, 0.01243408, 0.04481194, 0.02679387, 0.02645015, 0.01879817, -0.07565699, -0.03698091, -0.07884298, -0.02481815, 0.12647198, -0.11098626, 0.01720981, -0.06832145, -0.07677837, -0.07871960, 0.01663783, 0.08396139, 0.02812928, 0.02064915, -0.02262068, -0.09094704, 0.02631618, -0.08844335, 0.01816588, 0.08046541, -0.12307904, 0.02341716, -0.01640539, -0.02386260, -0.02857707, -0.01526381, 0.13831612, 0.26252928, -0.15323016, -0.01025706, 0.03846603, 0.07130742, 0.02223295, -0.07715549, -0.00866395, 0.28177291, -0.15750302, 0.01909994, 0.03600420, 0.02638853, 0.01036225, 0.11398090, 0.11497068, 0.36903241, -0.00852918, 0.08239851, -0.04451347, 0.05330950, -0.00439973, 0.20894340, 0.05279859, 0.34494978, -0.00876743, 0.05146833, -0.05019092, 0.02570513, 0.01945251, 0.18512219, -0.02211074, 0.21759832, -0.02714960, 0.00996702, -0.05460990, 0.09465240, 0.05409074, 0.13357756, -0.02898744, 0.14652997, 0.07163737, -0.05216889, -0.01359009, -0.00131855, 0.05746114, 0.18828350, 0.05162278, 0.08945771, 0.08946718, 0.03667324, 0.01433767, -0.00390752, 0.07192694, -0.07767697, 0.08588043, 0.10065221, 0.11015548, 0.04611788, 0.01725823, 0.11863018, 0.02716234
    .float 0.04316487, 0.05875150, -0.03536503, -0.10397357, 0.03793583, 0.00455914, 0.02607181, 0.11445593, 0.08824980, 0.10522413, -0.07208706, -0.09087998, 0.14147896, -0.00618873, -0.01546041, 0.14664923, 0.05530990, 0.08551922, -0.03659143, 0.06499965, 0.07517961, -0.00612743, -0.01116058, 0.06507752, 0.13942511, 0.14103845, -0.09395157, 0.05337703, 0.10305852, -0.03696490, 0.01343768, 0.00355765, 0.02579880, 0.05746264, -0.05582599, 0.22782463, 0.11538828, -0.05104564, -0.02063327, -0.02170088, 0.11343354, 0.06949565, -0.01358430, 0.22281781, 0.12879920, -0.01326629, 0.01298967, 0.10764575, 0.09513552, 0.07918409, 0.07967214, 0.25882727, 0.03540861, -0.01588832, 0.02604732, 0.05075727, 0.00282906, 0.08379716, 0.08282972, 0.17413718, 0.09381461, 0.02262731, 0.13593787, 0.06585757, 0.05949361, 0.06434366, 0.09633682, 0.17091288, 0.11711219, 0.01347096, 0.03505280, 0.04613328, 0.16164388, 0.09045062, 0.01096108, 0.12427498, 0.08704369, 0.06763946, 0.08689728, 0.12096331, 0.07218013, 0.03893721, 0.00153963, 0.11500929, 0.13749391, -0.00481467, 0.00390632, 0.09504596, 0.12942983, 0.02241706, -0.01635165, 0.10861611, 0.09790523, -0.11181582, 0.06734940, 0.11098193, 0.09273353, 0.01212136, -0.04030962, -0.06882731, -0.05707745, 0.06840035, -0.08868501, 0.00269584, 0.02093097, 0.03698843, -0.10835743, 0.01533677, 0.00182611, -0.02890404, -0.06825285, -0.05508691, -0.01084282, -0.03635710, -0.03805473, 0.15141016, 0.06806502, -0.00123279, 0.01121459, 0.00700960, -0.05377163, -0.14998703, -0.12085923, 0.14593641, -0.06704098, -0.05333821, 0.05372057, 0.03609069, -0.07461490, -0.10195991, -0.01185101, 0.15316413, 0.01398740, -0.05418336, -0.01808653, 0.00567455, -0.00849668, -0.07666455, 0.07648872, 0.21916477, -0.01120752, -0.14784141, 0.06696969, -0.00240482, 0.08352101, -0.01128740, 0.07365248, 0.12871501, -0.00056066, -0.20659496, 0.04668423, -0.02300028, 0.13813096, -0.02951138, 0.13897534, 0.08597898, 0.11782329, -0.21813093, 0.01344555, 0.05504306, 0.11106168, -0.01220000, 0.02390814, -0.04227398, 0.01825571, -0.21493661, 0.06216844, 0.12146349, 0.10742470, 0.10472713, 0.05875973, -0.01546176, 0.05166249, -0.12894797, 0.08193330, 0.07390799, 0.06214488, 0.15465580, -0.05264413, -0.09457478, 0.09377152, -0.05741203, 0.05206114, 0.17917602, -0.05859784, 0.07192480, 0.03972670, -0.00826890, 0.00746528, -0.14366128, 0.09360003, 0.11555694, -0.06463912, -0.12469146, -0.10032783, -0.12683122, -0.06465241, -0.01414563, -0.11739873, -0.09106900, -0.09399358, -0.13007449, -0.04577811, -0.07556643, 0.02802597, -0.01197169, -0.10278121, -0.12369527, -0.01140374, -0.10883869, -0.14095722, 0.19868931, -0.01743250, -0.09042262, -0.07616785, 0.01972394, -0.08504299, -0.09250783, -0.03592223, 0.27724588, -0.08910769, -0.04938448, 0.00161408, -0.12280396, -0.00617821, -0.09088782, -0.01937443, 0.16388577, -0.03222055, -0.24893226, -0.00429549, -0.06559161, 0.04123358, -0.21673347, 0.16516696, 0.20232552, -0.07660093, -0.26792061, -0.02356581, -0.10076803, 0.12740333, -0.09687333, 0.17226760, 0.01638827, 0.04358030, -0.17769817, -0.05065406, 0.01757837, -0.00993502, -0.04697178, 0.08749843, -0.08542710, 0.05991479, -0.21803324, -0.02519668, -0.01761185, 0.00400090, 0.00755944, 0.07981433, -0.18869491, -0.07229587, -0.11121868, 0.02596061, 0.04899173, -0.03461911, -0.02222253, -0.07094952, -0.12940103, 0.01573722, -0.00272811, -0.09919741, 0.00540776, 0.00853763, -0.01662638, -0.15222487, -0.10896904, -0.06064346, 0.01647807, -0.00769467, -0.07090711, -0.12887420, 0.09358958, -0.20993486, -0.04531888, -0.01099646, 0.00931546, -0.05874667, -0.04869769, -0.17969267, -0.14483832, -0.12206063, -0.06520404, 0.00440836, 0.04327853, -0.15866390, -0.09082232, -0.12791036, 0.00142093, -0.13757674, 0.03131402, -0.08974312, -0.07684258, 0.01989670, -0.01414797, -0.07044695, -0.01150636, -0.00155244, 0.24616349, -0.14169371, -0.07991195, -0.13754006, -0.10872174, -0.00210943, -0.11800874, 0.06802738, 0.21134511, -0.11582547, -0.04617837, -0.02684101, -0.02787744, -0.02868551, -0.08621110, 0.00580189, 0.22975148, 0.00122255, -0.12524800, 0.02980416, 0.02335646, 0.04791825, -0.21070167, 0.12333048, 0.09278940, -0.04309402, -0.18225645, -0.02737182, 0.04181282, 0.13970688, -0.20734754, 0.17635451, 0.08377408, -0.09157076, -0.24368632, 0.00559228, 0.02031706, 0.04366431, -0.05163312, 0.28776258, -0.03571802, 0.01859773, -0.17970760, 0.01043962, -0.11762207, 0.01913098, 0.02372531, 0.12806615, -0.24644452, -0.07828432, -0.04281756, -0.14737290, -0.12633592, 0.00404073, -0.05131568, 0.03923555, -0.19555114, -0.09543467, 0.05361536, -0.04655696, -0.15224665, -0.08333843, -0.02122594, -0.02614349, -0.15631557, -0.17902997, 0.08427209, -0.18386993, -0.11338600, -0.18794100, -0.08062939, -0.22995748, -0.13155006, -0.23418051, 0.09990408, -0.22609773, -0.16243725, -0.20095211, -0.17337452, -0.07635916, -0.08893818, -0.12519762, -0.05454107, -0.13130896, -0.15797351, -0.06690273, -0.06346512, 0.00011719, 0.10865082, -0.09488824, -0.04067252, -0.07397422, -0.11720316, -0.11271261, -0.08708841, 0.04599120, 0.22489397, -0.11273316, 0.03541164, -0.06260397, 0.00195198, -0.02185393, -0.11954229, 0.03397364, 0.19938469, 0.03164701, -0.04932706, -0.02474198, -0.01614490, -0.04499713, -0.04609783, 0.06779755, 0.04234686, 0.00565158, -0.22636168, -0.02324061, 0.00178446, 0.01675143, -0.11328495, 0.10207745, 0.08059754, -0.04200010, -0.28066835, 0.07729565, 0.01688991, 0.14401902, -0.14338617, 0.18855059, 0.07939889, -0.02771660, -0.44079337, 0.04582734, -0.04617035, 0.02422200, -0.07049833, 0.09757050, -0.18349735, -0.15208738, -0.23745340, -0.06613418, -0.17354311, -0.04762205, -0.08077955, 0.11366794, -0.36425051, -0.12293716, -0.09662957, -0.11085418, -0.10479774, -0.04210201, -0.02603350, -0.03357914, -0.25894022, -0.11943468, 0.10491437, -0.12028028, -0.12467034, 0.01666437, -0.07336850, -0.00675579, -0.11950406, -0.14705652, 0.07089679, -0.19929601, -0.06356533, 0.06427437, -0.23683940, -0.11560455, -0.06597450, -0.13418022, -0.17717281, -0.10637335, -0.20008367, -0.14697875, -0.09525340, -0.01944992, 0.00726902, -0.11494817, -0.11246969, -0.04381335, -0.02888853, -0.08094756, 0.03416187, -0.11313401, -0.07503008, -0.04887295, -0.08101816, -0.09268928, 0.02776386, -0.00129443, -0.01184824, 0.00399517, 0.32274115, 0.00463460, -0.08490004, -0.05862391, -0.02368805, -0.07155284, 0.03570358, 0.07502367, 0.22730400, -0.01093814, -0.11999159, -0.02249702, 0.00797441, 0.03897746, -0.06000443, -0.02552013, 0.07573619, 0.09090425, -0.07443824, 0.08300429, 0.00460651, 0.06429556, -0.14336760, 0.01037347, 0.02621370, 0.02835968, -0.29163671, 0.02793504, -0.00730833, 0.13636421, 0.06376807, 0.16038056, 0.02674226, -0.00869853, -0.25661445, 0.01622812, -0.07546242, 0.05641121, 0.06711426, 0.09868556, -0.28608233, -0.06507802, -0.11214487, 0.03648296, -0.13262700, -0.11481577, 0.08089834, -0.08476730, -0.26930469, -0.07050562, -0.05795500, -0.09196652, -0.09665472, -0.11685173, -0.05190936, -0.10247654, -0.21137226, -0.00486663, 0.06973599, -0.03743733, -0.03089439, 0.01203144, -0.03956765, -0.15348214, -0.13606919, 0.04117662, 0.09890608, -0.02237389, 0.03859608, -0.01006082, 0.02215626, -0.19116744, 0.06208207, -0.03586892, -0.27915454, 0.00302102, -0.03078052, -0.19485919, -0.12926379, -0.04178442, -0.10758793, -0.00181804, -0.12051463, -0.04451355, -0.13378000, -0.02835841, -0.01399312, -0.11811666, -0.02957560, -0.00911115, -0.10655063, -0.01284941, -0.03784673, -0.03768001, -0.07799751, -0.04017595, 0.27280819, -0.00176302, 0.01884820, 0.03950772, -0.02516196, 0.03672651, 0.01219383, -0.05232468, 0.14065880, -0.00106615, -0.01335482, 0.08222711, 0.02164021, -0.01210562, -0.07439235, -0.04597874, 0.02911845, 0.01099473, -0.02783752, 0.05736418, 0.03750866, 0.09269669, 0.04963424, 0.01305266, 0.19096316, 0.04862830, -0.22612861, 0.11925301, 0.02441665, 0.06200897, 0.08734951, 0.14613311, -0.02108299, 0.11869606, -0.04661981, 0.04345708, -0.00912691, -0.00196371, -0.00126007, -0.07096311, -0.14068820, -0.04031111, -0.02252297, 0.10692120, 0.07109410, -0.00942555, 0.04707509, -0.09572136, -0.07638252, -0.02305360, 0.08690568, -0.01851600, -0.02060246, 0.04224793, 0.02226698, -0.17548336, -0.01688914, -0.01447490, 0.04632995, 0.03084703, -0.00979320, -0.03419343, 0.05397822, -0.27656844, 0.04593717, 0.08506361, 0.04602924, -0.02673854, 0.11033859, -0.13659598, 0.02765748, -0.29679039, 0.11673632, 0.09567474, -0.05779013, 0.07789389, 0.11857080, -0.19101018, -0.15873043, -0.11183239, -0.05286344, -0.03393769, -0.15168428, -0.16357696, -0.18670073, 0.00807474, -0.01413346, -0.12356998, -0.14125015, -0.02214867, -0.07346999, 0.04103527, 0.03869906, -0.12722863, -0.03791755, -0.22805519, 0.13999613, 0.08715042, -0.10738984, 0.03488444, -0.03953222, 0.02339823, 0.01670953, -0.21357530, 0.22300138, 0.07646023, 0.02995498, 0.01002654, 0.09667836, 0.03269072, 0.02291628, -0.08961715, 0.14090934, 0.10797369, -0.03283043, -0.00496244, 0.09009896, -0.05615174, 0.00235858, -0.21304336, 0.05843802, 0.09741760, -0.18428577, 0.08508272, 0.00678232, 0.00233286, -0.04620206, -0.14778405, -0.10139019, -0.00684727, -0.10722020, 0.14813553, -0.02935442, 0.03538782, -0.00641095, -0.16874926, -0.09167303, 0.09280352, 0.03293786, -0.02186539, 0.00451849, 0.14266966, 0.07526834, -0.17278774, 0.09097002, 0.03404675, -0.04687470, -0.04257812, 0.07090539, 0.05836873, -0.01449136, -0.17762420, 0.07125392, 0.04389038, 0.01856566, 0.10980613, 0.03960661, -0.03617614, -0.03604776, -0.24122693, 0.02894899, 0.03454135, -0.00296204, 0.03105310, 0.01747453, -0.12152164, 0.05903807, -0.11749166, 0.08247446, 0.01796651, -0.05098521, 0.09500399, -0.01573586, -0.10515010, -0.15008026, -0.08521335, -0.11166972, -0.12155190, -0.12395394, -0.10438266, -0.17951351, -0.05012396, -0.10275880, -0.12447133, -0.12552808, -0.07526605, -0.03645326, -0.05376154, -0.01200164, -0.09030557, -0.06110134, -0.32189590, 0.12713243, 0.05891721, 0.00128612, 0.05487735, 0.07869860, -0.02184485, -0.01869705, -0.30945551, 0.12620085, 0.04664816, -0.01732538, 0.09538033, 0.10211588, 0.00374969, 0.05137986, -0.33658341, 0.13809119, 0.11315506, 0.01598690, 0.00596046, 0.08451725, -0.03034129, 0.04630724, -0.17272016, 0.06367461, 0.08017037, -0.09107185, 0.06380779, 0.13080084, 0.12053070, 0.09236996, -0.25161803, -0.06338809, 0.12075190, -0.07125573, 0.05546528, 0.06095507, 0.11979002, 0.02805327, -0.10856964, -0.04701584, 0.14054370, -0.08256784, 0.01756483, 0.02089642, 0.00966143, 0.03605738, -0.06858744, -0.02669857, 0.01448457, -0.00195801, 0.11860398, 0.04113454, 0.01463002, 0.02976352, -0.17848256, -0.03824544, 0.06008692, -0.03106853, 0.09683148, 0.06085424, 0.06767093, -0.04039520, -0.02716460, -0.06391584, 0.06361143, -0.10859156, 0.02608280, -0.06070835, -0.03685714, 0.00003046, 0.01621935, 0.06891151, 0.02731502, 0.01716227, 0.01066347, -0.07982523, -0.16818072, -0.14498644, -0.12305319, -0.09862352, -0.00537776, 0.00163463, -0.09674757, -0.08459363, -0.00366738, -0.12145804, -0.17491271, -0.08932804, -0.05623205, -0.10736237, -0.16120118, -0.03400259, -0.09207033, -0.03377567, -0.23282154, -0.06948993, -0.01637359, -0.06612148, -0.07592888, -0.00671639, -0.00756094, 0.01068465, -0.39381954, -0.12254362, 0.03893645, 0.07639226, 0.04890176, 0.14964874, 0.08058153, 0.14289181, -0.42391720, -0.00358579, 0.09817056, 0.09287113, 0.13373841, 0.07507403, -0.09857297, 0.17616004, -0.20889243, -0.08229767, 0.10520223, 0.08928774, 0.11466144, 0.07670685, -0.03283411, 0.04740873, -0.05150877, -0.09262543, 0.11510560, -0.01709015, 0.12834570, 0.03332093, -0.02374775, 0.07075468, 0.03727282, -0.03397032, 0.07401087, -0.10372727, 0.13519335, 0.08690313, 0.03308501, -0.03524929, -0.02350930, -0.12001115, -0.04505353, -0.03716556, 0.06672272, -0.02803125, 0.16206124, -0.06543969, -0.05795795, -0.02263122, -0.01975044, -0.10492805, 0.03104539, -0.06390681, 0.01566099, 0.00935917, -0.05415758, -0.02158128, 0.01419804, -0.13226853, -0.01786109, -0.09517789, -0.07180811, -0.06969912, -0.06166011, 0.00261009, 0.00510828, -0.08778043, -0.03745636, -0.03062822, -0.12928963, -0.07936180, -0.06064110, 0.00781917, -0.07528836, -0.00110657, -0.09449505, -0.02336292, -0.08390667, -0.12323441, -0.07623713, 0.00194644, -0.05327887, -0.06694023, -0.10292552, -0.04557813, 0.05370430, -0.05686785, -0.24015521, -0.15163840, 0.00919281, -0.03547941, -0.15450934, -0.04895111, 0.05750235, -0.02736169, -0.31233239, -0.22640805, -0.02908101, 0.02931738, -0.01713679, 0.02363403, -0.06459994, 0.09289058, -0.43827406, -0.12228148, 0.08617567, 0.13699563, 0.03445034, -0.00316061, -0.13608056, 0.07049191, -0.46080124, -0.19701575, 0.09531388, 0.25393060, 0.05824566, 0.06704069, -0.23194425, 0.01824219, -0.24054250, -0.16473816, -0.00658231, 0.07361315, 0.01340449, 0.00675265, 0.07313969, 0.03856115, 0.02386046, -0.06653040, -0.07841006, 0.04500201, 0.02896711, -0.03761757, 0.14913179, -0.00498283, -0.04974588, -0.01101797, -0.08476333, 0.05567143, -0.00249054, -0.04591727, -0.00845422, -0.09224266, 0.00170369, 0.00488548, -0.05140301, 0.02161794, 0.01490873, -0.11356308, -0.00689181, -0.07109334, 0.03009314, -0.00968231, -0.14693716, -0.06251422, -0.10625425, -0.13284878, -0.03701016, -0.15505673, 0.01111127, -0.07345389, -0.11859024, -0.11878498, -0.13364026, -0.08357829, -0.11519669, -0.00473926, -0.06959267, -0.03295193, -0.13782632, 0.00668523, -0.11451856, -0.09013318, -0.41616544, -0.17693104, -0.03990198, 0.05430760, -0.15278850, -0.16943838, -0.08425166, -0.14050563, -0.03842005, -0.09714441, -0.09148587, -0.01827895, -0.05214426, -0.19876069, -0.10362060, -0.04496423, -0.02908093, -0.16347583, -0.09815098, -0.09481382, 0.01367602, -0.05267588, -0.10505581, 0.00486718, -0.07916924, -0.07056817, -0.03452529, 0.01434431, 0.01428114, -0.05806575, -0.06047824, -0.06991033, -0.24602635, -0.03245040, 0.05796151, -0.11445716, 0.00785373, 0.04839411, 0.02929555, -0.11434197, -0.29804668, -0.07159585, 0.09592033, -0.15908366, -0.03051640, -0.07231464, -0.09618745, 0.00290814, -0.33612633, -0.10228177, -0.04784459, -0.19238523, -0.06720344, -0.04854276, -0.09272379, -0.05604706, -0.24039751, -0.12397242, 0.08623037, -0.09340786, -0.04949160, 0.01697250, 0.00288408, -0.03127089, -0.20948032, -0.13801716, 0.01377951, -0.01255491, -0.09499776, -0.07723432, -0.04877359, -0.03180711, -0.08516908, -0.09464943, 0.01401060, -0.03860110, -0.19038928, 0.00351412, -0.01413910, -0.17936626, -0.00823947, -0.14177649, -0.03575860, -0.11308050, -0.28135514, -0.12252104, -0.22564536, -0.30481797
    .float -0.01251051, 0.05059915, -0.03667144, -0.04137616, -0.08902609, 0.09971064, -0.06384879, -0.05495465, -0.06279175, -0.05830862, -0.03680178, 0.02919980, -0.13467199, -0.03067685, -0.12337525, -0.10736391, -0.09528451, 0.01601442, 0.07272428, -0.02877364, -0.22438206, -0.02042847, 0.00104624, -0.06229634, -0.17803739, -0.22958495, -0.05809258, -0.06506000, -0.18986587, -0.08595746, -0.12191949, -0.14874285, -0.17644759, -0.12310803, -0.04009974, -0.05669131, -0.10140885, -0.04369829, 0.00913084, -0.06165965, -0.09824280, -0.23374180, 0.01763392, -0.15266046, -0.17117403, -0.14597525, -0.03887993, -0.09170431, -0.05484284, -0.29332075, -0.03675705, -0.06047293, -0.13185047, -0.14008804, -0.11194169, -0.14768860, -0.07476774, -0.30256248, -0.09486642, -0.11498870, -0.00572254, -0.13738085, -0.04490856, -0.14020079, -0.17866366, -0.18834542, -0.29025313, -0.08222382, -0.02231234, -0.04340437, -0.06440563, -0.20891292, -0.22928774, -0.13096450, -0.22629400, -0.14266160, -0.21847798, 0.00699568, -0.10558128, -0.21519935, -0.27372748, -0.06278034, -0.07002497, -0.18598436, -0.18297049, 0.04075284, -0.20945616, -0.18777351, -0.29968217, -0.07190653, 0.04191419, -0.05132999, -0.23714727, 0.08046881, -0.18306410, -0.21786235, -0.02899249, 0.06614294, 0.06397302, -0.02889513, 0.00603672, 0.06407532, 0.10801935, 0.08640647, -0.01178978, 0.04852015, 0.00375682, 0.01507318, 0.02581654, 0.10230635, 0.15736920, 0.00499731, 0.04823075, 0.06958812, 0.12002110, 0.08742427, 0.06156739, 0.10068920, 0.01830924, 0.00079236, 0.03607983, 0.06803547, 0.12182207, 0.06639211, -0.04130117, 0.05935351, -0.01626669, 0.02333403, 0.02129894, 0.03750734, -0.00671145, -0.20079276, -0.03586032, 0.01900313, -0.05907592, -0.01817950, -0.01383780, -0.07778529, -0.01109918, -0.22203758, 0.00426386, -0.06869262, 0.02012707, -0.03575786, -0.09145920, -0.03343859, 0.00363826, -0.27637929, -0.02066861, -0.24646997, -0.03013354, -0.01245201, -0.04133093, -0.17021029, -0.01292809, -0.10063580, -0.11115667, -0.17259263, -0.05523392, -0.06102036, -0.07500979, -0.16941315, 0.05930749, -0.00802303, -0.11616269, -0.00227357, 0.01314361, -0.15060085, -0.03856030, -0.21938595, 0.02674507, -0.08138466, -0.03362481, 0.14692633, -0.10870965, -0.01095408, -0.07954896, -0.31490052, -0.01149761, 0.02497479, -0.09442085, 0.03968073, -0.11195221, -0.10215677, -0.08947328, -0.20326693, -0.01811787, -0.02232585, -0.05212390, 0.07680460, -0.15218230, -0.10412128, 0.13666761, 0.09482617, 0.02478326, 0.00813009, 0.05772425, 0.05934471, 0.11673413, 0.03153822, -0.02125165, 0.13399841, -0.01038379, 0.08981299, -0.00568258, 0.14990851, 0.15683056, 0.06672360, -0.06476944, 0.11230920, 0.07035534, -0.03107976, 0.00043706, 0.23954254, 0.16059349, 0.07243122, 0.01183911, 0.18887374, 0.00975482, 0.07432728, 0.03596695, 0.20975284, 0.03545080, 0.13339779, 0.00507056, 0.04162097, 0.04016237, -0.12212530, 0.04186263, 0.30948603, 0.03174946, 0.00093754, -0.03151282, 0.06404566, 0.02355506, -0.25580189, -0.07683085, 0.11945239, 0.04376519, -0.11112700, -0.01549223, 0.03482588, -0.00482129, -0.24849476, -0.03313329, -0.00511954, 0.00254894, -0.09603318, -0.04978038, -0.03029756, 0.12133362, -0.02172998, -0.01138172, -0.06196025, -0.10681967, -0.01702693, 0.10614403, 0.02252835, 0.12389362, 0.09494298, 0.01734967, -0.15509440, -0.05153298, -0.04145557, 0.08092876, -0.06304100, 0.12684005, 0.00627870, -0.05770475, -0.16984427, -0.05785407, -0.03173384, -0.00987364, -0.18776730, 0.03849364, 0.03995531, 0.01429032, -0.08445694, -0.08787877, -0.01822013, -0.04763284, -0.23964858, -0.03024585, -0.03969363, -0.01165390, -0.00769846, -0.05634001, -0.14421123, 0.05460120, 0.11549143, 0.10481280, -0.00260422, 0.07365540, 0.09942629, 0.02966167, 0.12117945, 0.03243911, 0.10313807, -0.03439180, -0.06939388, 0.03040787, 0.07046378, 0.05323650, 0.12063521, -0.03617180, 0.20508365, 0.00178384, -0.22491157, 0.04745322, 0.19805266, 0.07108063, 0.11317138, -0.02124089, 0.08842215, 0.01574452, -0.20521200, 0.10506784, 0.18262330, 0.09078974, 0.04531233, -0.09193253, 0.03990771, -0.05272440, -0.30491492, 0.02332004, 0.12420714, 0.12796889, 0.09187047, -0.10428360, 0.05579921, -0.05530791, -0.23128140, 0.02528972, 0.07189050, 0.10286774, 0.06881017, -0.10142490, 0.10830490, 0.00103967, -0.15861975, 0.02195671, 0.11702982, 0.02660209, 0.01769663, 0.01967222, 0.06730650, 0.05775423, 0.04323318, 0.07315063, 0.10619375, 0.07784426, 0.06868125, -0.01448032, 0.05460818, 0.03875998, 0.13182162, 0.02952023, -0.01734443, -0.01555588, 0.06022092, 0.04173663, 0.00010320, 0.12186573, 0.08968543, 0.08834220, -0.20476800, 0.09861724, -0.01202269, -0.03115830, -0.07078058, 0.09416624, 0.09313931, 0.07691074, -0.18581127, 0.09211366, -0.14167732, -0.02128990, -0.01593809, 0.03756096, 0.01861356, -0.03356783, -0.12571484, 0.04299338, -0.17223991, 0.11986447, 0.18303850, 0.06243430, -0.06084673, 0.11467440, 0.08711231, 0.06647366, 0.16062038, 0.04756919, 0.09251966, -0.01895894, -0.26339188, 0.07005946, -0.01043004, 0.09125769, 0.00860542, -0.02613159, 0.08077426, -0.05365355, -0.36543831, 0.09593215, 0.10132871, 0.08351587, 0.03349216, 0.01031950, 0.10098031, -0.02806982, -0.34496847, 0.09349444, 0.09900393, 0.09978536, 0.01823642, -0.10716823, -0.02279560, 0.08012713, -0.20070979, 0.09881205, 0.01801275, 0.07628857, -0.05045805, -0.22012684, -0.00293553, 0.12961665, 0.03725841, 0.05307321, 0.08196418, -0.04372854, 0.00289062, -0.23133646, 0.09947698, 0.11876784, -0.00393978, 0.05248227, 0.17242420, 0.07209885, 0.03625550, -0.06664979, -0.01528264, 0.02613828, 0.12623933, 0.10786612, 0.06418595, 0.04179458, 0.11415917, 0.00464687, 0.08390552, 0.16884883, 0.10822549, 0.00316949, -0.05787044, 0.08394639, 0.02496805, -0.02195332, -0.03676660, 0.16369289, 0.05380132, -0.00472283, -0.14947307, 0.00643629, 0.02020257, -0.10581623, 0.06076503, 0.01139393, 0.06356247, -0.02207381, -0.12078688, 0.03623935, -0.10307629, -0.07598659, -0.08702894, -0.03788637, -0.09170036, -0.04765101, -0.17651685, -0.01818052, -0.17882611, 0.11675876, 0.17041661, 0.06951656, -0.05501731, 0.05406722, 0.19879167, 0.07371851, 0.13312048, 0.01098895, 0.04826597, -0.00461521, -0.16081947, 0.11040122, 0.04777138, 0.07266776, 0.09200473, 0.09881613, 0.03494092, 0.11073233, -0.38081577, 0.08645549, 0.10063154, 0.02189079, -0.05139472, -0.05315424, -0.04936289, 0.11093058, -0.17899077, 0.03843913, 0.09572876, -0.04588563, 0.00686414, -0.28219169, -0.21556237, 0.22856455, 0.06080106, 0.00709195, -0.04338305, -0.08851060, -0.06711975, -0.26987424, -0.14261703, 0.10925733, 0.23986442, 0.00332717, -0.08532274, -0.04106825, -0.12817535, -0.11885560, -0.10967037, 0.14128304, 0.10730457, 0.00047718, 0.12019472, -0.05295916, -0.01702139, 0.03343452, -0.01429524, -0.03390463, 0.08404254, -0.00638353, 0.02412706, 0.03326145, 0.05771340, 0.03915827, -0.09334285, -0.01224276, 0.01761748, 0.05955618, 0.01628313, 0.01820998, 0.07946391, -0.02369521, 0.02220607, 0.10151479, -0.01237483, 0.02526707, -0.01291451, 0.04146987, 0.05554673, -0.01900541, 0.04916636, 0.04446583, -0.03486433, -0.04575670, -0.03513590, -0.00640374, -0.01922287, 0.10687704, -0.10961109, 0.05550147, -0.04621882, -0.11289386, -0.08097539, -0.11849557, -0.00860937, 0.12124934, 0.02984572, 0.01314694, -0.02965244, 0.04127226, 0.17684200, -0.00342267, 0.04818540, 0.02157760, 0.02100508, 0.07395329, -0.07115584, 0.05712348, 0.10513362, 0.06738057, 0.08493975, 0.02522421, -0.07389243, 0.19961533, -0.02413992, 0.04655896, 0.07317989, 0.00748386, -0.06972216, 0.01658872, -0.14708374, 0.37325472, 0.25761586, -0.09605344, -0.00428902, -0.05531330, -0.02009436, -0.05132975, -0.13604774, 0.36747491, 0.25820166, -0.16057070, -0.14449976, -0.04805058, -0.08851775, -0.20349336, -0.14191727, 0.15067200, 0.25578150, -0.18810175, -0.29621571, -0.09414822, -0.10612357, -0.02744115, -0.23316975, 0.11021812, -0.00621180, -0.02331476, -0.13325636, -0.05032074, 0.03313762, -0.01521098, -0.01612133, -0.09628574, -0.06120405, 0.03879554, -0.03873475, 0.00453919, 0.02856234, -0.02819508, 0.11937302, 0.10091819, 0.01588380, 0.10653932, 0.04394976, 0.13790207, -0.03386703, 0.09629095, 0.14897889, 0.09648862, 0.10768943, -0.03478903, -0.07418024, 0.01426712, -0.04404510, 0.07543410, 0.08296019, -0.05721162, 0.07450690, 0.00212720, -0.10717738, 0.06727961, -0.00789394, -0.06234870, -0.00145374, -0.01096588, 0.04368872, 0.00848690, -0.02244256, 0.02009943, -0.03926613, 0.00723454, -0.04930554, 0.01396737, 0.04768151, -0.00264707, 0.03519455, 0.07840360, 0.06492676, -0.05736609, -0.06828935, -0.04178805, 0.00399857, -0.03300660, -0.01011129, -0.10576171, -0.09231348, 0.04673965, -0.15188502, 0.22340624, 0.18207264, -0.13068032, -0.11966794, -0.06642549, -0.03499859, -0.00502184, -0.14102493, 0.22679564, 0.40995398, -0.15472701, -0.18031323, -0.06649001, -0.16726796, -0.09770851, -0.12442635, 0.14479753, 0.33000508, -0.16784850, -0.28717172, -0.08937141, -0.23479643, -0.14115037, -0.17634508, 0.07031996, -0.08385411, -0.10541660, -0.19448914, -0.14150821, -0.15149429, -0.08515031, -0.12843230, 0.13258570, -0.08336981, 0.05051916, -0.18381734, -0.10909669, -0.04665759, -0.07961652, 0.03136810, 0.09485292, -0.12716669, 0.02224682, -0.10181230, 0.11697396, 0.02284392, -0.09017870, 0.09237646, 0.15525068, -0.04782150, 0.05828063, 0.02192861, 0.04948846, 0.09710339, 0.09541224, 0.05799734, 0.03140888, 0.08285205, 0.05689903, 0.17752960, 0.02380069, 0.10406759, 0.03223672, 0.10262302, 0.02339801, 0.03581996, 0.00426583, 0.06365888, -0.02716988, -0.00028954, -0.10322071, 0.04620491, 0.06235715, -0.07177228, -0.05966213, -0.01711412, -0.04793374, -0.15282804, -0.02965318, -0.07487874, -0.09566471, -0.04921928, 0.02090399, -0.02494189, -0.11680228, 0.00298053, 0.01432426, -0.15681131, 0.00806695, 0.00373378, -0.03929800, -0.00009960, -0.02379986, -0.02315633, 0.00580348, -0.12636687, 0.01851265, 0.15311511, -0.11294014, -0.11090620, -0.15984593, -0.07968003, 0.06588251, -0.14169608, 0.00373938, 0.21239009, -0.12065095, -0.18886937, -0.19704124, -0.07700075, -0.05917703, -0.13468321, -0.02874793, 0.12682441, -0.06409261, -0.27600238, -0.06397653, -0.11650939, -0.03975649, -0.01088060, -0.02595812, 0.06889915, -0.02535671, -0.12666447, -0.07089439, -0.02373640, -0.12390178, -0.12117431, 0.06637299, 0.10949201, 0.10372569, 0.01335525, -0.02267700, 0.02542485, -0.15518411, 0.01066696, 0.07926694, 0.08686748, 0.05613423, -0.04117900, 0.03861721, 0.03678520, -0.05766218, 0.01719416, 0.04955067, -0.06538728, -0.06518936, 0.14110784, 0.01675339, 0.02878266, 0.02475670, -0.02671417, -0.02820528, 0.02181962, -0.06382323, -0.02989717, 0.01824326, -0.06971678, -0.12843855, -0.02875490, 0.01069944, -0.02238085, -0.01253495, -0.01977527, 0.05308697, -0.09861089, -0.10861924, 0.01487358, 0.00804825, -0.12462186, -0.11330500, 0.03129059, -0.04777665, -0.18800966, -0.09354606, -0.09206345, -0.06200010, 0.09341735, -0.03914179, 0.00566864, 0.05222645, -0.04783424, -0.09731867, -0.06018889, 0.11242956, 0.17874292, -0.17050806, -0.04156008, -0.07746336, -0.10948795, -0.00721706, -0.15052268, 0.04994516, 0.16661069, -0.11363649, -0.24017085, -0.08113047, -0.16160449, -0.05936325, -0.24041474, 0.01065592, 0.11284271, -0.14139383, -0.35702667, -0.13528313, -0.12707032, -0.04336483, -0.13193801, -0.09349260, 0.14393966, -0.10864563, -0.28099921, -0.17872560, -0.08488339, 0.03183645, -0.11959989, -0.06312868, 0.23715137, -0.03449472, -0.05557549, -0.02692098, -0.06514008, -0.13167548, -0.04991978, 0.01678637, 0.13025929, -0.05524921, -0.06124881, -0.00538735, -0.01836927, -0.19556069, -0.02037040, -0.06677373, 0.09500275, -0.10908940, -0.05894739, -0.04656124, -0.09486993, -0.21196584, -0.15208679, -0.12512349, -0.00597852, -0.06402864, -0.02292787, -0.04986692, -0.06964915, -0.31538707, -0.11677444, -0.04006536, -0.05232012, -0.02073707, -0.08654651, -0.04560073, -0.02029018, -0.20933671, -0.12804605, 0.01787786, 0.03070971, -0.13241391, -0.00368542, -0.01980907, -0.21416035, -0.09500574, -0.15413728, 0.10396358, -0.07546088, -0.19199817, 0.02617131, -0.12239984, -0.13478820, -0.14343801, -0.04516759, 0.13019432, -0.01985642, 0.01265512, -0.09014487, 0.02765141, 0.03128143, 0.07159033, 0.01650461, 0.05778170, 0.10548145, 0.01777681, -0.15469787, 0.01966470, -0.02254237, -0.03450503, -0.12443260, 0.10080494, 0.06261613, -0.16354324, -0.26973489, -0.08514634, -0.04567402, -0.02546540, -0.13988213, 0.01634037, 0.27194092, -0.07032626, -0.30932033, -0.03092946, -0.10055724, -0.04337863, -0.23460987, -0.05416436, 0.36019406, -0.05416435, -0.17246529, -0.08967829, -0.01478142, -0.09105401, -0.11665788, -0.04514535, 0.23364581, -0.04266921, -0.14019683, 0.00194982, -0.05892160, -0.08141102, -0.02301759, -0.11779026, 0.17668955, -0.10502204, 0.01027643, 0.05196008, 0.03129613, -0.15211594, -0.10123369, -0.07836249, 0.14306606, -0.00500718, 0.01408164, 0.03048259, -0.07055283, -0.23379594, -0.02539618, -0.09233709, 0.05446423, -0.04909679, -0.12468717, -0.00636164, -0.06099420, -0.36630157, -0.11884497, 0.03703173, -0.04916722, -0.13774705, -0.17147127, -0.03813136, -0.13485509, -0.18051675, -0.20749627, -0.05431157, -0.07657339, -0.10481083, -0.18397287, -0.12332997, -0.15108454, -0.29778755, -0.19285114, 0.09723672, -0.09312614, -0.12571599, -0.02940939, -0.20791996, -0.10257145, 0.11031250, 0.00591888, 0.12167972, -0.03933417, 0.03114586, 0.01259453, -0.09868389, -0.09296690, 0.05879266, 0.00049356, 0.06404459, -0.00028360, 0.03014172, 0.03816469, 0.04039805, -0.01323281, -0.05195809, 0.04715010, 0.00768783, 0.17734469, 0.03818993, 0.00286288, 0.10364640, 0.01394862, 0.03922946, -0.00042270, 0.03007396, 0.22296081, -0.07155530, -0.14387946, -0.02408181, -0.03311126, 0.09288977, -0.03459078, -0.01175706, 0.23615812, -0.06943899, -0.16572854, 0.05573298, -0.05364605, 0.04271468, 0.05763866, 0.01338675, 0.29790020, 0.06962597, -0.14212333, 0.09434255, -0.04200274, 0.13303261, 0.07201441, 0.06257675, 0.23250432, 0.05615956, -0.00263248, 0.05018691, 0.02313964, 0.04746009, -0.01202828, 0.01872948, 0.14273587, 0.08823322, -0.06459551, 0.06902850, -0.00337639, -0.09343942, -0.01999667, 0.04740078, 0.07157160, -0.03940631, -0.09497194, 0.08787045, -0.06497798, -0.15638879, -0.05865734, 0.04883923, 0.11994594, 0.06899344, -0.20454609, 0.01097435, -0.15624960, -0.03998321, -0.13429128, 0.04226015, 0.03150596, -0.11417062, -0.11449253, -0.03324338, -0.18053968, -0.14323878, -0.15937948, -0.01191338, -0.12774473, -0.13072990, -0.15020996, -0.13937493, -0.12459321
    .float -0.17298071, -0.19537207, -0.06180979, -0.12942363, -0.15971352, -0.03923952, -0.09277927, -0.18838462, -0.11744507, 0.01682583, -0.06519527, -0.05821035, -0.10113396, -0.10408910, 0.03789605, 0.03256102, 0.01079965, 0.04843131, -0.03972869, -0.12255985, -0.07842680, -0.03271193, -0.01331170, 0.01842926, -0.03217235, -0.04544348, 0.09125319, -0.05220924, 0.01009186, 0.00656838, 0.01885082, 0.07757495, 0.04197977, -0.02697675, 0.03029092, -0.11438446, 0.08762670, 0.02456757, 0.09989382, 0.05748465, 0.04211899, -0.07814714, 0.05778844, 0.01190615, 0.02066923, 0.00483508, -0.08179069, 0.04273444, 0.07183103, 0.05262241, 0.03427591, 0.08236785, 0.02592542, 0.06211038, 0.02763655, -0.06123812, -0.02161383, 0.04694950, 0.02441711, 0.06541098, -0.00784626, 0.08686657, -0.02710323, -0.04827826, 0.03891996, 0.05256254, 0.02759394, 0.02485445, 0.03621245, -0.02421303, -0.03441886, 0.12293172, -0.01891601, 0.05119519, 0.04414301, -0.02121757, 0.02033770, -0.00550283, -0.10412939, -0.02188212, -0.00078391, 0.06777252, -0.09670845, -0.04394897, 0.10181930, -0.03143243, 0.06778148, 0.03040515, -0.08367632, 0.02613372, -0.05513347, 0.02016381, -0.04323481, -0.03008228, 0.05090173, 0.02421709, -0.19777508, -0.02193087, -0.13658850, -0.03854100, -0.07956202, -0.03503482, 0.00969006, -0.08278226, -0.03753192, -0.10131966, 0.03427323, -0.01887923, -0.02874337, -0.13210292, 0.01323434, -0.03580096, 0.04808236, -0.06997970, 0.02459158, -0.05638589, 0.04747231, -0.11245506, -0.04945057, 0.00560051, 0.03606926, -0.04249945, -0.03856527, -0.01621954, 0.00548446, -0.15986702, -0.08796730, -0.00044134, 0.00825546, 0.00272448, -0.04920061, 0.00186519, -0.01464862, -0.03857970, 0.03428750, 0.06585266, 0.05313366, 0.02690321, 0.08490780, -0.04472664, 0.08110826, -0.07370210, -0.00391277, -0.02406367, 0.02151076, 0.09227176, 0.08355533, 0.01564979, 0.04336340, -0.15985319, 0.07969210, 0.07370435, 0.04770646, 0.03543160, -0.06832042, -0.00240692, 0.04395192, 0.06034059, -0.00569864, 0.00363712, 0.03459015, 0.05717374, -0.00130704, 0.03412096, 0.07037364, 0.04124555, 0.04030187, 0.03950145, 0.08920173, 0.05951778, -0.10247415, 0.03082922, -0.02308572, 0.00103397, 0.04822868, 0.02229941, -0.07514353, 0.02410507, -0.10810103, 0.00027501, 0.03402066, 0.03610121, -0.03638569, -0.00864469, -0.02935838, 0.10334592, 0.05645787, -0.06344242, -0.03563353, 0.05075630, -0.04206694, 0.06755979, -0.09475805, -0.04941579, -0.03912550, -0.03781809, -0.03538913, -0.12497012, -0.07068289, -0.00235212, -0.04038939, 0.02025113, 0.00031051, 0.01732846, 0.05956523, -0.09825923, -0.08231386, -0.07507569, 0.05703261, 0.03681251, -0.03423824, -0.07200015, -0.06585135, -0.09780203, -0.09329777, -0.05438811, 0.04576740, -0.03612605, 0.00032488, 0.04572570, -0.06556866, -0.09283879, 0.02944099, -0.04706984, 0.11356459, 0.00146893, -0.07295967, -0.18619224, 0.04076162, 0.00358462, -0.00298703, 0.02748419, 0.07708500, -0.01821370, -0.04295979, -0.17228338, 0.08288097, -0.02889175, -0.00224102, 0.02451301, 0.11025349, 0.08240031, -0.05021852, -0.21812299, 0.02255855, -0.00301586, 0.00880790, -0.01238924, 0.03951710, 0.01260813, -0.05763657, -0.12996148, -0.03813050, 0.08564637, 0.02800038, 0.06883167, 0.07131611, 0.02414962, -0.05324860, -0.01041679, 0.05982799, 0.00582611, 0.03973779, -0.03169633, 0.01462184, 0.01250016, -0.01152929, -0.00397049, -0.02203741, 0.06212804, -0.05250571, -0.00549214, 0.04250059, 0.05547335, -0.05511250, -0.03807596, 0.08852063, 0.30092439, 0.08775062, 0.02218672, 0.05772243, 0.03537559, -0.05392428, 0.05172654, 0.05811008, 0.00881810, 0.10653252, 0.08635657, 0.03412305, -0.07669398, -0.07639278, -0.03233676, -0.04277444, -0.06972961, -0.03702283, -0.01290872, 0.06545988, -0.01276768, -0.01463688, -0.09274019, 0.03844341, -0.03150337, 0.00605267, 0.02392124, 0.11258351, 0.05127015, 0.03083541, 0.00063669, 0.01573217, -0.13086990, 0.04090345, 0.03997238, 0.00038410, 0.05905381, -0.03691021, 0.06303857, 0.00247898, 0.02148507, 0.01404913, -0.06046026, 0.08835276, 0.07840698, -0.09888564, 0.01783129, -0.04512222, 0.01939149, -0.05095636, -0.01273155, 0.13396533, 0.01267736, -0.09594981, -0.08627361, -0.01618903, 0.09755451, 0.01333700, -0.04681412, 0.15354873, -0.05921244, -0.14129776, -0.11385356, -0.05673648, 0.05218515, -0.07002891, 0.00061385, 0.18689719, 0.07975608, -0.16730717, -0.13937360, -0.00147040, 0.06687543, -0.02999857, 0.03135338, 0.22009669, 0.05856276, -0.11355654, 0.00855836, -0.04682541, 0.11633463, -0.00313502, 0.04320759, 0.10410026, 0.09957759, -0.15823634, 0.06419339, 0.01155456, -0.00446393, 0.08211844, 0.04096247, 0.12075115, 0.02642024, -0.10097737, 0.01057607, 0.00156855, 0.06103559, 0.07274712, 0.16217890, 0.03968616, 0.03681903, -0.13957047, 0.02490285, 0.03935870, -0.03749242, 0.09607783, 0.16942723, -0.03701496, -0.01452236, 0.01724756, -0.06806862, 0.00098206, -0.08935236, 0.01316840, 0.07703879, -0.05361160, 0.01999660, -0.00718673, -0.10908556, 0.00445381, -0.08795220, 0.06003763, -0.01116885, -0.01202700, 0.13146965, -0.00930513, 0.04501719, -0.03155184, -0.13132223, 0.02641900, 0.07837620, 0.06958363, 0.05260082, -0.19963405, -0.02452477, -0.08332862, 0.03922206, 0.08987365, 0.03186189, -0.08162981, 0.13408107, -0.25180787, 0.01965347, -0.07998759, 0.05219107, 0.07715322, -0.02282958, -0.04788060, 0.07789982, -0.31819880, -0.07915095, -0.02425790, 0.27336192, -0.02666692, 0.06425307, 0.08389788, -0.04338710, -0.24388848, -0.07110519, -0.00750973, 0.08697333, -0.01678945, -0.01693434, 0.19669940, 0.00515261, -0.04526071, -0.07112835, -0.01975595, -0.00795924, -0.02620996, -0.00913884, 0.26037645, -0.06004009, 0.10347734, -0.05728927, 0.12986402, 0.02102953, -0.03635536, 0.03832803, 0.30025122, 0.05704565, -0.02946774, 0.08553548, 0.10225964, -0.01545287, -0.02345080, 0.09054047, 0.29926184, 0.16678429, -0.11234964, 0.01908020, 0.03895285, 0.09624064, -0.00757957, 0.10763764, -0.03344449, 0.06560759, -0.13232632, 0.08849160, 0.08203585, -0.13157591, 0.16243616, 0.10192010, 0.07617520, -0.03550830, -0.10031471, -0.06788723, 0.03160993, -0.06538767, 0.01195996, -0.03264126, -0.07891887, 0.08697473, -0.12254494, -0.11392132, 0.04222601, -0.00575567, -0.02318452, 0.00080509, -0.11048550, 0.08348261, -0.06855033, -0.13383022, 0.09065068, 0.04934256, 0.05946531, 0.02270723, -0.10194253, 0.17325170, -0.26989830, -0.09286019, -0.02810925, 0.06934206, 0.05588821, 0.11956549, -0.05972599, 0.21927917, -0.37129965, -0.05128291, 0.07297488, 0.08133351, 0.09714598, 0.02409133, 0.05523850, 0.11756759, -0.34458953, -0.04355937, 0.02483752, 0.26083320, 0.06239593, 0.07738517, 0.10071506, -0.01985414, -0.31231171, -0.16256832, 0.05621954, 0.01693515, 0.04326041, 0.04515237, 0.17628010, -0.08828299, -0.04057574, -0.07264531, 0.00848037, -0.15433422, 0.06372948, 0.03182410, 0.24152620, -0.01021394, 0.04267418, -0.06290610, 0.08185413, -0.09890172, 0.03094634, -0.02235468, 0.23376200, 0.09876394, -0.00008556, -0.07538444, 0.03915446, -0.04004482, -0.06890184, -0.01235248, 0.16229959, 0.04074040, -0.17212208, -0.13178939, -0.03091344, 0.00022079, 0.04880381, 0.07573518, -0.05350773, 0.09011722, -0.17064381, 0.04675046, 0.03514756, 0.02061572, -0.02119225, 0.09098401, -0.05420054, -0.08583212, -0.14595200, -0.05775253, 0.06692544, 0.00250274, -0.14036466, -0.06764342, 0.01224166, 0.02518593, 0.03751764, -0.07357637, 0.00074268, -0.00337284, 0.06437880, 0.05323763, 0.07515824, 0.04979253, 0.04589657, -0.12471129, 0.05430030, 0.10112695, 0.00222007, -0.01478735, 0.09965538, 0.07441733, -0.05563683, -0.42121652, 0.05400092, -0.02865503, 0.07143840, -0.00010894, 0.08940322, 0.06528130, -0.13092145, -0.09678369, 0.11843768, 0.05281416, -0.02762469, 0.10635748, 0.13018990, 0.05395487, -0.20934165, -0.00926165, 0.10444354, 0.17101415, 0.12826857, 0.08436064, 0.03296866, 0.02074781, -0.11413965, -0.05236038, 0.06321742, -0.08009801, 0.10004977, -0.02633341, 0.06556282, 0.03529757, -0.07966495, -0.11234313, -0.06317825, 0.00778548, -0.03704811, -0.05865277, 0.14725052, -0.01971233, -0.12043723, -0.20266777, -0.13764682, 0.07472237, -0.08414392, -0.06204070, 0.06431922, -0.04852935, -0.16806459, -0.09543961, -0.11582828, 0.15970728, -0.07165425, -0.07882697, 0.06059738, -0.02133598, -0.11866469, -0.12317745, -0.11536300, 0.18646149, -0.06014512, -0.07780760, 0.11082207, -0.00449643, -0.07325941, -0.05587695, -0.13746041, 0.04693210, -0.05921784, -0.03506829, -0.12079468, -0.08968387, -0.09589685, -0.00530054, -0.10512885, -0.09818707, -0.04044320, -0.05844929, -0.05038351, -0.07867719, -0.06159084, 0.00632824, -0.04018503, -0.03389298, -0.12904075, -0.06698614, 0.06758150, -0.09460972, -0.05127246, 0.05243823, 0.01635554, -0.01489779, -0.01881323, -0.05882007, 0.08900296, -0.13536149, 0.00440259, -0.04145057, 0.05545790, -0.10948637, 0.02826268, -0.00771940, 0.12322288, 0.02943173, -0.03844027, 0.18175404, 0.07067494, -0.01616787, 0.03221646, 0.15409510, 0.20811814, 0.08115572, -0.11097011, -0.04402314, 0.07774696, 0.03695888, 0.15414663, 0.08643257, -0.06565861, 0.02880178, -0.24408706, -0.05845315, -0.06656007, 0.03964842, 0.07779208, -0.02894040, -0.13155966, 0.04259484, -0.22547317, -0.02847730, -0.10388997, 0.11882219, -0.04029845, -0.02249397, -0.06553721, 0.00823094, -0.22712232, -0.05542462, -0.11346351, 0.08738069, -0.09170895, -0.09885895, 0.00341896, 0.00419841, -0.14116523, -0.11467713, -0.11953589, 0.12652662, -0.13806359, -0.09283837, 0.14182983, -0.12983570, -0.05518341, -0.07551827, -0.08122708, 0.04098158, -0.03295303, -0.06557965, 0.11326306, -0.07317992, -0.08968773, -0.08546405, -0.00901892, -0.18061678, 0.01201574, 0.01457176, -0.17811823, -0.15139607, -0.10720109, -0.03882547, -0.07619128, -0.17208016, -0.14264019, -0.17388515, -0.00219544, -0.04337848, -0.10809306, -0.04184744, -0.02838987, -0.01926084, -0.03979042, -0.03787175, 0.04065645, -0.11344180, -0.09552580, 0.15746593, 0.01837202, -0.17441389, -0.04449514, -0.10931589, 0.09175114, -0.16410725, -0.04560450, 0.25959268, 0.00423013, -0.12385190, -0.04970802, 0.01937757, 0.20682155, -0.07710519, 0.01236737, 0.20364372, 0.09314902, -0.04186118, 0.08030517, 0.02798992, -0.02334785, 0.05860560, -0.14554186, 0.12818372, 0.01737529, 0.04882200, 0.13228482, 0.05695764, -0.12903161, 0.09462184, -0.18925203, -0.00473098, -0.02317600, 0.10065255, 0.07292585, -0.03725588, -0.07858034, -0.03202261, -0.15731525, 0.03654543, 0.00366046, 0.12292963, -0.04578413, -0.09044039, 0.07744794, 0.01087129, -0.17331994, 0.10198138, -0.09094610, 0.10390804, -0.10604164, -0.05297525, 0.09561057, -0.06796895, -0.18068983, 0.04507781, 0.04744330, -0.03348156, 0.02060047, -0.08892674, 0.08518112, -0.09332582, -0.16672054, 0.00644712, -0.03532811, -0.05640378, -0.04737792, 0.02465822, -0.19267097, -0.06735663, -0.17135726, 0.09932905, 0.12814984, -0.26081261, 0.03372858, -0.02231095, -0.07940652, -0.08237881, -0.09831144, -0.04085301, -0.14325477, -0.15501434, -0.07119568, -0.04911191, -0.04602730, -0.07769285, -0.07301626, -0.04843830, -0.05771213, -0.12920471, -0.03845518, 0.01460364, 0.04620633, -0.00206113, -0.18801802, 0.06603720, 0.01209613, -0.07858748, 0.02809679, -0.03655061, 0.03811680, -0.01164980, -0.24300529, 0.28356114, -0.10903104, -0.10711478, -0.04726881, -0.05271682, 0.17084613, -0.06207344, -0.26383144, 0.07611898, 0.02278780, 0.02274304, 0.05537430, 0.01641544, 0.17325982, 0.07482219, -0.22379020, -0.00390118, -0.05077977, 0.01041400, 0.00864764, -0.03079816, 0.09996723, 0.03410713, -0.14077833, -0.05455516, 0.04375923, -0.07488804, -0.00583766, -0.12489777, 0.11776967, 0.06185603, -0.07366867, 0.05311552, 0.05250682, 0.01376317, 0.01342401, 0.00913095, 0.08976815, -0.03877519, -0.09252920, 0.00938369, -0.03908414, -0.01637652, 0.05475834, -0.05841645, 0.00341252, 0.06293858, -0.18443464, 0.07312335, 0.04848732, -0.12944932, -0.05485249, 0.00960733, -0.06799082, -0.03019548, -0.20227943, 0.02121462, 0.00582789, -0.10468899, 0.06850576, -0.01337904, -0.14677511, 0.10107798, -0.09152218, 0.10122164, 0.07908373, -0.02498831, 0.10012567, -0.01367615, -0.00666292, -0.07321064, -0.06369492, -0.00472306, -0.03206038, -0.16520369, -0.12537691, -0.11351053, 0.08648606, -0.00697603, -0.21699448, -0.11256477, 0.07588830, 0.01055695, -0.05319404, -0.06073111, -0.01699485, 0.10162980, -0.26407653, -0.08669397, 0.06563486, -0.09838666, -0.02102634, 0.04656948, -0.05817351, 0.10387105, -0.23920815, -0.32785624, -0.03889827, -0.00534120, 0.00662095, -0.06379756, -0.01413392, 0.04201097, -0.39771482, -0.30366302, 0.01676465, 0.00814684, 0.02849665, -0.02318673, 0.11445282, 0.03909255, -0.40828213, -0.26986459, 0.00157910, 0.04268667, 0.08801816, 0.06880986, 0.17334402, 0.14336167, -0.28006780, -0.04760781, 0.02702873, -0.07213110, 0.07258185, 0.03908622, 0.09844194, 0.00396154, -0.08732323, -0.06208005, -0.02812859, -0.08884159, -0.00786587, -0.01223231, 0.11867467, -0.03803508, -0.03859070, -0.08067878, -0.03482786, -0.08854637, 0.04182210, 0.02323991, 0.16588393, -0.03730430, -0.07575038, 0.02148319, 0.07693962, -0.04552660, 0.02001313, 0.06101923, -0.12822437, 0.08130164, -0.08131576, 0.02222897, 0.06559241, -0.00655902, 0.06576452, 0.05609363, -0.05653423, 0.09571842, -0.10450871, 0.02817489, -0.02078929, -0.06387518, -0.01629156, -0.10327120, -0.17415857, -0.16859142, -0.10369223, -0.04478684, -0.03741967, -0.10806070, -0.08253124, -0.06272907, -0.06559607, -0.07967348, -0.15691255, -0.09478132, -0.05156743, -0.03926986, -0.15111451, -0.06916592, -0.11329919, -0.09807170, -0.23220271, -0.25434992, 0.00514949, -0.01512626, -0.04627272, 0.00421121, -0.14182547, 0.10679542, -0.34879079, -0.24346954, -0.02940429, 0.06971124, -0.00662587, -0.00466471, -0.02020645, 0.09586578, -0.36153993, -0.16099340, 0.06030159, 0.11929157, -0.00530560, -0.01433538, 0.04062419, 0.03262776, -0.55701673, -0.02876830, -0.01573036, 0.03244521, 0.05756408, 0.04809899, 0.09060260, 0.05097899, -0.32382971, -0.04554103, 0.00571917, 0.04248121, -0.00592956, 0.04149425, 0.15634683, 0.00585336, -0.25349620, -0.02929202, -0.01680219, 0.00050563, 0.02522622, 0.05961672, 0.06537909, 0.00445384, -0.13707499, -0.07513365, -0.02944088, 0.00009143, -0.00185154, -0.03932924, -0.09322011, -0.02514172, -0.08665275, -0.11101299, 0.01095261, -0.02662711, -0.02564035, 0.00942990, -0.23182219, -0.03694338, -0.04269051, -0.02028992, -0.06354263, -0.03505861, 0.00682224, -0.03768134, 0.01041936, -0.12066542, -0.08176581, -0.10935868, -0.14038958, -0.03320598, -0.13563465, -0.18573651
    .float 0.02311948, -0.03109706, -0.01307843, -0.03243149, -0.04749588, -0.02693855, -0.10183541, -0.11478540, -0.07197893, -0.04642748, -0.10370918, -0.07238369, -0.09187455, -0.02768981, -0.06625167, -0.06519129, -0.07324588, -0.10806641, 0.00494123, 0.01272584, -0.15026459, -0.09811531, -0.03572473, -0.16331343, -0.20835002, 0.06959037, 0.03759940, -0.05075003, -0.22027344, -0.02324414, 0.07510190, 0.04913970, -0.06698254, -0.05340128, 0.05318504, -0.10753238, -0.06158059, -0.01243832, 0.01222889, -0.01113622, -0.01372973, -0.28424996, 0.03805141, -0.18784086, -0.01711357, -0.14156300, 0.00920888, -0.03187279, -0.09621496, -0.42742530, -0.05583522, -0.19557822, 0.00879459, -0.27253756, -0.09669641, -0.12214816, -0.11664107, -0.28691736, -0.10225729, -0.17268978, -0.09706726, -0.13784087, -0.09620318, -0.12321444, -0.10799786, -0.29038960, -0.10639642, -0.25356302, -0.17188342, -0.11675978, -0.18705153, -0.26072350, -0.11927186, -0.19452855, -0.06603183, -0.05896648, -0.13117176, -0.07087179, -0.18683797, -0.22237086, -0.23313090, -0.10674348, -0.14348686, -0.06748442, -0.10338945, -0.04697025, -0.18566534, -0.19665362, -0.34942800, -0.07638083, -0.08981353, -0.09124801, -0.19660948, -0.00661588, -0.21847247, -0.25849423, -0.14292623, -0.18393174, -0.07427590, -0.01465790, -0.12092654, -0.00142802, -0.16046555, -0.25601503, -0.20142445, -0.08542602, -0.10454915, -0.03283872, -0.11617392, -0.08732961, -0.07355919, -0.08744217, -0.18019618, -0.10008626, 0.08361468, -0.12924302, -0.00426808, -0.06717633, 0.06873810, 0.04632507, 0.01083535, -0.08072983, 0.04796669, -0.23716526, 0.02335010, -0.09852559, 0.01078337, 0.00916171, 0.06299098, -0.01680935, 0.10150199, -0.32398379, 0.04266908, -0.00412321, 0.05422471, 0.09120719, 0.06394621, -0.03177324, 0.03117498, -0.17573042, 0.00197738, -0.01156406, -0.02087555, 0.01650088, 0.01659116, 0.06968138, 0.09794604, -0.10040168, 0.00276664, 0.03768124, -0.09343150, 0.03520174, 0.04927608, -0.06987824, 0.06929724, -0.12858936, -0.04346139, -0.05289018, -0.03819687, 0.02819237, -0.04021014, 0.02514706, -0.01882696, -0.14615540, -0.03187425, -0.11883948, -0.04045736, -0.07335239, -0.08733828, -0.20755294, -0.07303510, -0.10638802, -0.08891424, -0.20832819, -0.16248932, -0.11682308, -0.06305262, -0.27753273, -0.11328064, -0.11713255, -0.02736722, -0.01690236, -0.10003156, -0.19790463, -0.11867111, -0.15278640, 0.01169136, -0.10919879, -0.07859147, 0.02457521, -0.14009111, -0.18477954, -0.27410024, -0.06036716, -0.10923745, -0.01204511, -0.19442903, -0.05738929, -0.16163862, -0.16466229, -0.14163829, -0.06336032, -0.00705108, 0.05389458, -0.05338004, -0.11094349, -0.10434249, -0.14245589, 0.03234481, -0.10506839, 0.02723431, -0.07075506, -0.04935219, -0.07941682, -0.09674238, -0.07893564, 0.12125780, -0.03148991, 0.08127496, -0.26498178, 0.06035723, -0.25395530, 0.02794270, -0.00741654, 0.09475925, -0.06040890, 0.12831438, -0.46303442, 0.03792707, -0.21446101, -0.01730124, -0.00227771, 0.16442102, 0.13270263, 0.03360248, -0.30511147, 0.05036230, -0.06212858, 0.09158238, 0.07651914, 0.09637947, 0.15051295, 0.06251734, -0.33734748, 0.03174352, 0.10670307, 0.03485438, 0.03632228, 0.12865381, 0.03508324, -0.06652495, -0.16763875, -0.05487555, 0.06742243, 0.06510949, -0.00007763, 0.04846055, 0.07845189, -0.15179627, -0.11297669, -0.00046226, 0.04406842, -0.07587892, 0.00276495, -0.01000271, 0.03519554, -0.19003242, -0.06583209, -0.00976704, -0.00803056, -0.08459032, -0.04319555, -0.14977972, -0.00455887, -0.16604163, -0.04955687, -0.02303386, -0.15365548, -0.14767496, -0.10589473, -0.19039318, -0.20850137, -0.14216851, -0.06672672, -0.07485056, -0.11828241, -0.10584085, -0.17652436, -0.16860729, -0.02221382, -0.05931159, 0.09780642, -0.07544640, -0.05964076, -0.03943870, -0.15364197, 0.06362523, -0.09074174, 0.03019410, 0.09393737, 0.01318898, -0.08156578, -0.08927175, 0.00529263, 0.04015692, -0.10560845, -0.00148080, -0.00933424, -0.00456457, -0.07727425, 0.03044470, -0.04710754, 0.12976371, -0.00819953, 0.03865093, -0.02875248, 0.03674363, -0.12220363, 0.04345240, -0.05159058, 0.10157260, 0.01229065, 0.10349945, -0.07762568, 0.00528211, -0.06673574, 0.08672541, -0.01357740, 0.23724563, -0.02431334, 0.06260045, -0.18897805, 0.05203432, 0.02149278, 0.01759093, 0.08021326, 0.19836648, 0.07577414, -0.08168885, -0.22212671, -0.00049126, 0.04528716, 0.07906021, 0.02643120, 0.03707441, 0.04637234, -0.26805827, -0.09971769, -0.00244385, 0.01827989, 0.05709995, 0.02158331, 0.09008961, 0.00557997, -0.25496677, -0.14430159, -0.06302492, 0.14326772, -0.07057650, -0.08093075, -0.03263938, -0.05274461, -0.44741735, -0.06820862, -0.09299600, 0.12888113, 0.01718246, 0.01703285, -0.18769358, -0.00120942, -0.30626863, 0.08163689, -0.02328442, 0.11918039, -0.08041236, 0.01527831, -0.20490558, -0.08935121, -0.05647419, 0.05579482, 0.00585660, -0.08930686, -0.01084638, -0.19073115, -0.13137205, -0.11456971, 0.06050294, 0.10715499, -0.07280510, -0.13628545, -0.06224578, -0.04792183, 0.09807577, -0.04991491, 0.03522563, 0.14823918, -0.01177382, -0.00345595, -0.06753448, -0.05472968, 0.06708183, -0.00916740, -0.07563401, 0.14235827, 0.00050571, -0.11458402, 0.01188007, -0.07128239, 0.07778970, -0.00104870, -0.00930182, 0.22150782, -0.03755262, -0.09357302, 0.02648286, -0.02179577, 0.13593780, 0.05251042, 0.00501228, 0.02385486, 0.04711778, -0.01745167, 0.01027030, 0.09679887, 0.16065821, 0.09275098, -0.06098119, -0.08848662, 0.03814481, -0.00802564, 0.04400294, 0.04921461, 0.23420793, 0.00613354, -0.01577516, -0.16022582, -0.03600591, -0.05953520, 0.00385496, 0.06857702, 0.12161539, 0.05524831, 0.09792413, -0.09071439, 0.04167135, -0.12208896, 0.03572457, 0.08938047, 0.00413550, -0.05173471, -0.08360706, -0.00014237, 0.11139556, 0.08441860, -0.01132878, 0.01622270, -0.02708452, 0.10162318, -0.27287382, -0.02068731, 0.01550540, 0.02923362, -0.03178325, -0.00298804, -0.06761569, -0.08640731, -0.16589096, 0.11292269, -0.01304312, -0.04951263, 0.03686408, 0.01731623, -0.16360341, -0.11573608, -0.07534301, 0.08940688, 0.06019593, -0.12104294, 0.08369762, -0.06011989, 0.03032035, -0.08663759, -0.02486135, 0.02347986, -0.06681617, -0.05378650, -0.09224361, -0.00511018, -0.01558027, 0.07469840, -0.10677885, 0.06917997, -0.03407432, -0.02887098, 0.00693588, -0.01637684, -0.04842567, 0.10852446, -0.13350646, 0.14757247, -0.08667786, -0.12356688, 0.05888413, -0.03096112, -0.03178576, 0.02817754, -0.20632136, 0.11760747, 0.00645830, 0.05348239, 0.09757283, 0.01304728, 0.00635982, 0.02117434, -0.37607619, 0.06159829, 0.01384375, 0.08574876, 0.02924288, 0.01258508, 0.10788871, 0.08460657, -0.14820935, -0.14210311, -0.00053687, 0.03314582, 0.00980194, 0.01681609, 0.14713815, 0.02536944, 0.06651278, -0.29129407, 0.08028544, -0.03597984, -0.04595427, 0.08771847, 0.03556306, 0.02928185, 0.02498101, 0.10245759, 0.12097269, 0.04500134, 0.03179042, 0.15140916, -0.00649780, 0.05776870, 0.17002770, 0.09876338, 0.07019188, 0.01123509, 0.11761598, 0.10345161, -0.06304645, 0.02717460, -0.16454285, 0.07459900, 0.13350098, -0.04319186, 0.08008139, 0.07497636, -0.15903921, -0.05680894, -0.21854123, 0.18497150, 0.07474125, -0.15684758, 0.15472852, -0.03231341, -0.40632051, -0.03489565, -0.05457399, 0.06489109, -0.05633048, -0.07886476, 0.02734205, -0.03026362, -0.05140986, 0.04154059, -0.11436365, -0.08707459, -0.07152583, -0.09881623, -0.00094194, -0.04364817, -0.08635678, 0.04643125, -0.13641034, 0.04055981, 0.03506165, -0.05520177, -0.04193201, 0.08644309, -0.18226516, 0.10915296, -0.21234706, 0.15662171, -0.01102689, -0.06051479, 0.00244505, 0.09440374, -0.04358633, 0.04814941, -0.45487636, 0.05497636, -0.02583282, 0.12598585, 0.07275435, 0.09683641, -0.06854253, 0.07950236, -0.44893736, -0.09494242, 0.03112831, 0.21099932, 0.08289450, 0.07227255, 0.00915975, -0.01427886, -0.11020690, -0.27354839, 0.01894269, 0.09877690, 0.02173117, 0.01864092, 0.00950317, 0.04701122, -0.05677246, -0.12618564, 0.00688511, 0.17989612, 0.02341142, 0.01030546, 0.03509944, -0.02262659, 0.04051611, 0.08583117, 0.08858825, 0.09231609, 0.01701190, 0.02539937, -0.04808326, -0.09976314, 0.20893070, 0.08238640, 0.09282620, -0.09324063, 0.02001100, 0.01682655, -0.13614428, -0.06447306, -0.05780301, 0.05526437, 0.01229998, -0.13087766, 0.11229331, 0.04938610, -0.23987541, -0.02554586, -0.07887194, 0.16134310, -0.00344686, -0.23594418, 0.06130322, 0.04608714, -0.38732269, -0.07480014, -0.01800007, -0.02079098, -0.14226140, -0.06012099, 0.00334225, -0.23788649, -0.01773798, -0.09226321, -0.12531678, -0.03162775, -0.02302670, -0.11964867, -0.10824395, -0.00482730, 0.01256738, 0.05272048, -0.18109198, -0.10759983, 0.03900751, 0.05764846, -0.03449557, -0.02852383, -0.01199815, 0.06924587, -0.26141775, -0.06913945, 0.00229714, 0.09166550, -0.02820497, 0.07007617, -0.03279977, 0.11218457, -0.29187721, -0.10162290, 0.03704286, 0.11475587, 0.03540919, 0.13566513, -0.17142674, 0.10942364, -0.25466749, -0.30927759, 0.04210170, 0.10555788, 0.10973448, 0.07468417, -0.01871867, 0.02214100, -0.01158388, -0.17244244, 0.00450634, 0.01605283, 0.07411957, -0.05804035, 0.14289910, -0.04163215, -0.03394947, -0.06896351, -0.06181996, 0.11953098, -0.15914385, -0.00507063, 0.09136190, -0.16157566, 0.11869591, 0.05409244, 0.05577433, 0.12699686, 0.01869084, 0.03504740, -0.04768506, -0.06158876, 0.09226000, 0.07406861, -0.03595677, -0.12143897, 0.03549399, -0.06299063, -0.10229271, -0.12849124, -0.05029548, 0.00189657, -0.00215891, -0.24307063, -0.03180020, -0.10655763, -0.34794077, -0.15358232, -0.07689837, 0.09849470, -0.11454834, -0.20355439, -0.05422863, -0.09919615, -0.29800919, -0.16682826, 0.03022348, -0.03775438, -0.21327129, -0.09136304, -0.15091382, -0.19554000, -0.07438068, -0.04960742, 0.02617557, -0.03661833, -0.11365918, -0.02087863, -0.08619397, 0.02811221, -0.05955730, 0.05689731, -0.10398099, -0.15005362, 0.05228269, 0.01879873, -0.08070192, -0.04570993, 0.11901975, 0.08055908, -0.12638398, -0.08754332, 0.05106410, 0.21053466, -0.03219752, 0.09292988, -0.03837592, 0.12948684, -0.02472951, -0.31145534, 0.13316809, 0.18949620, 0.02028941, 0.02750232, -0.14106990, 0.04658425, -0.08461615, -0.33799681, -0.06812992, 0.16255799, -0.03824779, -0.04468312, -0.06550165, -0.03200735, 0.02778932, -0.12134804, -0.17949526, 0.07373594, -0.09398605, -0.14428353, 0.09787036, -0.10615763, 0.01655972, 0.03411977, -0.08756791, 0.03177790, -0.07446916, 0.03927751, 0.04662003, -0.13477539, 0.01157646, 0.10517415, -0.01966270, -0.04065310, -0.09233253, -0.03744319, -0.03533591, -0.13990901, 0.05933930, -0.02282131, 0.00560023, -0.12852830, 0.01629815, -0.01491094, -0.18593486, -0.15447991, 0.11455848, 0.04852187, -0.03678901, -0.23357575, -0.05640759, 0.00657841, -0.30848148, -0.11637532, 0.04157929, 0.03975043, -0.10397527, -0.19738635, -0.07398862, -0.04510219, -0.01242001, -0.16535281, -0.01321044, -0.08674280, -0.15305234, -0.22610737, -0.02001411, -0.13053413, -0.00199672, -0.02815671, -0.04759217, 0.02548963, -0.06816479, -0.08604225, -0.22762799, -0.10858096, 0.01907437, 0.01660956, -0.05891339, -0.08362211, -0.00188182, -0.03000421, 0.01402263, 0.04264176, -0.01096135, 0.10656219, 0.09746972, -0.04170679, 0.09849139, 0.10627232, 0.08198085, 0.01010966, 0.02294133, 0.03230379, 0.10961456, -0.15168571, -0.03205280, 0.02475240, 0.03442747, 0.04868191, -0.08017656, -0.00690975, 0.07507208, -0.09159056, -0.16103916, 0.03561218, -0.06326061, -0.04758479, -0.03258774, -0.10121278, 0.24616176, 0.05419444, -0.11736980, -0.07957904, -0.03995821, -0.07771639, 0.02568471, -0.23364255, 0.18849584, 0.01506171, -0.16440597, -0.05883599, -0.08306108, -0.03365372, -0.05441085, -0.09479492, 0.07453989, 0.03451498, -0.10523923, -0.09008492, -0.04760209, -0.02013604, -0.11398751, -0.13433544, -0.04512221, -0.00042731, -0.12162311, -0.10094888, -0.12841129, -0.12374394, -0.13292481, -0.14691448, -0.08204453, -0.00865443, -0.02242135, -0.03553296, -0.00511259, -0.07328247, -0.09348325, -0.03438516, 0.00243984, -0.07944311, 0.00820805, -0.07476836, 0.03334585, -0.04241767, 0.07975001, 0.00878120, 0.06371741, -0.04548775, 0.00235095, -0.00881789, -0.02903686, 0.05325409, 0.01167516, -0.02123576, 0.02842294, 0.03775343, -0.08854863, -0.01678209, -0.04904953, 0.02315641, 0.03494174, -0.06024129, 0.08630501, 0.07711664, -0.08862406, 0.00033689, 0.01406300, -0.01873709, 0.09453794, -0.06797399, 0.03547700, 0.00489711, -0.04263445, 0.00457209, 0.02119800, 0.03308290, 0.01910087, -0.04363929, 0.07914582, 0.04117360, -0.09668249, -0.06840491, -0.04072398, -0.06383902, -0.06992070, -0.02127112, 0.08824766, 0.09267887, -0.09252945, -0.09253766, -0.01576758, -0.11580733, 0.02881374, -0.09819515, 0.06146836, 0.05941192, -0.09530398, -0.15585773, -0.05831396, -0.13758494, -0.10131784, -0.14132312, 0.12667876, 0.05181422, -0.11789738, 0.00904677, -0.03153415, -0.07803448, -0.12125488, -0.03765677, -0.00538543, 0.06611155, -0.01841703, 0.00241003, -0.04478364, -0.10323427, -0.20756352, -0.07954220, -0.05335408, 0.02745368, -0.03521574, -0.00770849, -0.04096195, -0.06621171, -0.13673639, -0.02822303, -0.13337485, -0.02949799, 0.06451034, 0.02259463, -0.05480215, -0.05535991, -0.08289808, -0.00689626, -0.05850463, 0.01271640, 0.16852798, -0.00714835, 0.02703409, 0.01122844, -0.06109297, 0.00204222, -0.01867135, 0.04184569, 0.02593401, 0.05656118, 0.01624355, -0.00191319, -0.09830040, 0.08385943, -0.02999121, -0.05200651, -0.00746361, -0.05977517, 0.07687405, 0.05504239, 0.03321349, 0.06737409, 0.06225584, 0.00017874, 0.05693239, -0.00447787, -0.02485486, 0.07359121, 0.03730544, 0.03058340, 0.18873860, 0.04884478, -0.02092236, -0.00938166, -0.00703672, -0.00052580, 0.02579502, -0.03775093, 0.12394063, 0.16595310, -0.06516599, -0.03149557, 0.01711215, 0.02792389, -0.00768583, -0.06960075, 0.14211071, 0.13489188, -0.00721313, -0.10114614, 0.02369120, -0.00048831, -0.09105813, -0.06369139, 0.19244269, 0.06738210, 0.00369819, -0.20073855, -0.03283027, -0.05626089, -0.09013693, -0.05772084, 0.12105618, 0.10338750, 0.05724168, 0.00553879, 0.05435571, 0.00686700, -0.17702173, 0.02038726, -0.03649054, 0.10963574, 0.06412663, 0.02908863, -0.00979436, 0.01126727, -0.11505425, 0.16670147, -0.03926297, 0.10138188, 0.09111030, 0.02401619, 0.09958473, 0.05519066, -0.00411078, 0.09297877, -0.04252132, 0.06384901, 0.11478446, 0.04556846, 0.05458238, 0.05778482, -0.03246354, 0.15705024, -0.04602613, 0.14889784, 0.25836697, 0.00011126, 0.21606947, 0.08284305, -0.09535757, 0.17627670, -0.07226867, 0.02890054, 0.16561098, 0.07904427, 0.03232881, 0.17253172
# 10 Dense biases input_matrix

#.global bias

## DENSE_BIAS BEGIN
dense_bias:
    .float -0.03106286, 0.03705791, 0.02269139, -0.03029724, -0.06035529, 0.07614812, -0.01340246, 0.00783046, -0.04537206, 0.00310953  # dense_bias[0:10]
## DENSE_BIAS END

#.global weights_size
Dense_weights_size: .word 11520
#.global biases_size
Dense_biases_size: .word 10


## Convolutional Layer Filters
#.global conv_filters

## FILTER BEGIN
# 5x5x1x8 Convolution Filter Weights
conv_filters:
filters:

    .float 0.35365000, -0.25632870, -0.44216833, 0.15843540, 0.09459874
    .float -0.54049528, -0.53184646, 0.03936517, 0.22009230, 0.37529960
    .float -0.35489005, 0.16881239, 0.40112746, 0.17117818, 0.06382002
    .float 0.15382232, 0.29604542, 0.19350106, 0.09646993, -0.35931459
    .float 0.21975663, 0.19576961, -0.10448381, -0.33259654, -0.34001398

    .float 0.08571938, 0.05246818, -0.13640058, -0.30890834, -0.36421886
    .float 0.06875563, 0.19719137, 0.11737578, 0.13695860, -0.21291991
    .float 0.21034351, 0.21524154, 0.37388074, 0.27672431, 0.17929377
    .float -0.18414134, -0.05535920, 0.13654919, 0.15569174, 0.04428325
    .float -0.26554105, -0.37985030, -0.30674666, -0.06591699, 0.15563157

    .float -0.47668391, -0.68407094, -0.70520109, -0.67043883, -0.46325693
    .float -0.72809625, -0.43222347, -0.49419975, -0.20195770, 0.06970464
    .float -0.70800102, -0.18568230, 0.10436120, 0.14102277, 0.26564130
    .float -0.32397175, 0.24313423, 0.43465859, 0.32206115, 0.22685176
    .float 0.00072064, 0.22624791, 0.38215321, 0.27373978, 0.38310599

    .float 0.40437764, 0.33044094, 0.03563903, -0.31558770, -0.67343509
    .float 0.29323167, 0.44412926, 0.12348457, -0.46441600, -0.72814035
    .float 0.32780445, 0.55394143, -0.00372886, -0.63345051, -0.58525175
    .float 0.50805616, 0.35701025, -0.22724269, -0.44985130, -0.51506984
    .float 0.49758628, 0.26255652, -0.46577427, -0.67713857, -0.07636263

    .float -0.06677402, 0.28758940, -0.03167812, 0.12183458, 0.16945498
    .float 0.17899673, -0.06593972, 0.15439457, 0.20059849, 0.25264758
    .float -0.03822786, 0.10297344, 0.25269485, 0.23477674, 0.24010961
    .float 0.21410686, 0.17173386, -0.00920129, 0.03571059, 0.17112833
    .float 0.23710646, 0.04126829, 0.01437632, -0.06402557, -0.16725379

    .float 0.24873076, 0.36437228, 0.46133071, 0.13680366, -0.03131076
    .float -0.02074433, 0.18580025, 0.49284437, 0.43374825, 0.28763518
    .float -0.53097165, -0.25913554, 0.25136501, 0.22662309, 0.08513539
    .float -0.55595225, -0.40321118, -0.33037508, -0.06057755, -0.04403410
    .float -0.46095756, -0.44417024, -0.10690765, -0.11406448, 0.02481145

    .float 0.20681159, 0.07821785, 0.19547315, -0.11196055, -0.03033486
    .float 0.04702268, 0.16883086, 0.00807513, 0.05254307, -0.05307937
    .float 0.18144301, 0.20721419, 0.05342988, -0.08593198, -0.13714716
    .float 0.21955657, 0.01897647, 0.07796083, 0.13668203, 0.22695531
    .float 0.22727366, 0.02078438, -0.06539664, 0.10971940, 0.20793316

    .float -0.13139348, 0.14177722, 0.00198559, 0.23046632, 0.11296314
    .float -0.09989197, 0.16594028, 0.12997375, 0.17657888, -0.11027593
    .float 0.11890251, -0.05423643, -0.08952732, 0.10709745, -0.02722906
    .float -0.15112159, -0.13366716, 0.19311699, 0.30353940, 0.27984601
    .float -0.03344458, 0.19415346, -0.04058857, -0.02419681, 0.16725861

# FILTER END

#.global conv_biases

## FILTER_BIAS BEGIN
filter_bias:
    .float -0.00493938, -0.01209035, 0.15486301, 0.07827792, -0.02660563, -0.00411384, -0.08101545, -0.00514472  # filter_bias[0:8]
## FILTER_BIAS END

#.global conv_filters_size
conv_filters_size: .word 200
#.global conv_biases_size
conv_biases_size: .word 8

## INPUTS BEGIN
#.global input_image

input_matrix:
# Image is Scaled between 0 and 255

## INPUT_MATRIX BEGIN
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.329412, 0.725490, 0.623529, 0.592157, 0.235294, 0.141176, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.870588, 0.996078, 0.996078, 0.996078, 0.996078, 0.945098, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.666667, 0.203922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.262745, 0.447059, 0.282353, 0.447059, 0.639216, 0.890196, 0.996078, 0.882353, 0.996078, 0.996078, 0.996078, 0.980392, 0.898039, 0.996078, 0.996078, 0.549020, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.066667, 0.258824, 0.054902, 0.262745, 0.262745, 0.262745, 0.231373, 0.082353, 0.925490, 0.996078, 0.415686, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.325490, 0.992157, 0.819608, 0.070588, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.086275, 0.913725, 1.000000, 0.325490, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.505882, 0.996078, 0.933333, 0.172549, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.231373, 0.976471, 0.996078, 0.243137, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.521569, 0.996078, 0.733333, 0.019608, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.035294, 0.803922, 0.972549, 0.227451, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.494118, 0.996078, 0.713725, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.294118, 0.984314, 0.941176, 0.223529, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.074510, 0.866667, 0.996078, 0.650980, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.011765, 0.796078, 0.996078, 0.858824, 0.137255, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.149020, 0.996078, 0.996078, 0.301961, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.121569, 0.878431, 0.996078, 0.450980, 0.003922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.521569, 0.996078, 0.996078, 0.203922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.239216, 0.949020, 0.996078, 0.996078, 0.203922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.474510, 0.996078, 0.996078, 0.858824, 0.156863, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.474510, 0.996078, 0.811765, 0.070588, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
## INPUT_MATRIX END
# Old file

# .data
# ## ALL DATA IS DEFINED HERE LIKE MATRIX, CONSTANTS ETC inner_loop_end
# # hello

# ## DATA DEFINE START
# .equ MatrixSize, 5
# matrix2:
#     .float -10.0, 13.0, 10.0, -3.0, 2.0
#     .float 6.0, 15.0, 4.0, 13.0, 4.0
#     .float 18.0, 2.0, 9.0, 8.0, -4.0
#     .float 5.0, 4.0, 12.0, 17.0, 6.0
#     .float -10.0, 7.0, 13.0, -3.0, 160.0

# ecall




# DENSE LAYER

# # These values are for testing only. Replace them with .float actual values.

# y = W.x + b

# .data

# x:    .float 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152            # x[1152]

# # W[10x1152]

# W:    .float 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152
# .float 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304
# .float 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456
# .float 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4427, 4428, 4429, 4430, 4431, 4432, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538, 4539, 4540, 4541, 4542, 4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608
# .float 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723, 4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810, 4811, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895, 4896, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912, 4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4931, 4932, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960, 4961, 4962, 4963, 4964, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5067, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094, 5095, 5096, 5097, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 5143, 5144, 5145, 5146, 5147, 5148, 5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5188, 5189, 5190, 5191, 5192, 5193, 5194, 5195, 5196, 5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5207, 5208, 5209, 5210, 5211, 5212, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 5220, 5221, 5222, 5223, 5224, 5225, 5226, 5227, 5228, 5229, 5230, 5231, 5232, 5233, 5234, 5235, 5236, 5237, 5238, 5239, 5240, 5241, 5242, 5243, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5251, 5252, 5253, 5254, 5255, 5256, 5257, 5258, 5259, 5260, 5261, 5262, 5263, 5264, 5265, 5266, 5267, 5268, 5269, 5270, 5271, 5272, 5273, 5274, 5275, 5276, 5277, 5278, 5279, 5280, 5281, 5282, 5283, 5284, 5285, 5286, 5287, 5288, 5289, 5290, 5291, 5292, 5293, 5294, 5295, 5296, 5297, 5298, 5299, 5300, 5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308, 5309, 5310, 5311, 5312, 5313, 5314, 5315, 5316, 5317, 5318, 5319, 5320, 5321, 5322, 5323, 5324, 5325, 5326, 5327, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 5348, 5349, 5350, 5351, 5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381, 5382, 5383, 5384, 5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5394, 5395, 5396, 5397, 5398, 5399, 5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 5409, 5410, 5411, 5412, 5413, 5414, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449, 5450, 5451, 5452, 5453, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5463, 5464, 5465, 5466, 5467, 5468, 5469, 5470, 5471, 5472, 5473, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5504, 5505, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5569, 5570, 5571, 5572, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591, 5592, 5593, 5594, 5595, 5596, 5597, 5598, 5599, 5600, 5601, 5602, 5603, 5604, 5605, 5606, 5607, 5608, 5609, 5610, 5611, 5612, 5613, 5614, 5615, 5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5693, 5694, 5695, 5696, 5697, 5698, 5699, 5700, 5701, 5702, 5703, 5704, 5705, 5706, 5707, 5708, 5709, 5710, 5711, 5712, 5713, 5714, 5715, 5716, 5717, 5718, 5719, 5720, 5721, 5722, 5723, 5724, 5725, 5726, 5727, 5728, 5729, 5730, 5731, 5732, 5733, 5734, 5735, 5736, 5737, 5738, 5739, 5740, 5741, 5742, 5743, 5744, 5745, 5746, 5747, 5748, 5749, 5750, 5751, 5752, 5753, 5754, 5755, 5756, 5757, 5758, 5759, 5760
# .float 5761, 5762, 5763, 5764, 5765, 5766, 5767, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 5781, 5782, 5783, 5784, 5785, 5786, 5787, 5788, 5789, 5790, 5791, 5792, 5793, 5794, 5795, 5796, 5797, 5798, 5799, 5800, 5801, 5802, 5803, 5804, 5805, 5806, 5807, 5808, 5809, 5810, 5811, 5812, 5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5821, 5822, 5823, 5824, 5825, 5826, 5827, 5828, 5829, 5830, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5838, 5839, 5840, 5841, 5842, 5843, 5844, 5845, 5846, 5847, 5848, 5849, 5850, 5851, 5852, 5853, 5854, 5855, 5856, 5857, 5858, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866, 5867, 5868, 5869, 5870, 5871, 5872, 5873, 5874, 5875, 5876, 5877, 5878, 5879, 5880, 5881, 5882, 5883, 5884, 5885, 5886, 5887, 5888, 5889, 5890, 5891, 5892, 5893, 5894, 5895, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 5910, 5911, 5912, 5913, 5914, 5915, 5916, 5917, 5918, 5919, 5920, 5921, 5922, 5923, 5924, 5925, 5926, 5927, 5928, 5929, 5930, 5931, 5932, 5933, 5934, 5935, 5936, 5937, 5938, 5939, 5940, 5941, 5942, 5943, 5944, 5945, 5946, 5947, 5948, 5949, 5950, 5951, 5952, 5953, 5954, 5955, 5956, 5957, 5958, 5959, 5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969, 5970, 5971, 5972, 5973, 5974, 5975, 5976, 5977, 5978, 5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011, 6012, 6013, 6014, 6015, 6016, 6017, 6018, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6057, 6058, 6059, 6060, 6061, 6062, 6063, 6064, 6065, 6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085, 6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099, 6100, 6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110, 6111, 6112, 6113, 6114, 6115, 6116, 6117, 6118, 6119, 6120, 6121, 6122, 6123, 6124, 6125, 6126, 6127, 6128, 6129, 6130, 6131, 6132, 6133, 6134, 6135, 6136, 6137, 6138, 6139, 6140, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6163, 6164, 6165, 6166, 6167, 6168, 6169, 6170, 6171, 6172, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6180, 6181, 6182, 6183, 6184, 6185, 6186, 6187, 6188, 6189, 6190, 6191, 6192, 6193, 6194, 6195, 6196, 6197, 6198, 6199, 6200, 6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6212, 6213, 6214, 6215, 6216, 6217, 6218, 6219, 6220, 6221, 6222, 6223, 6224, 6225, 6226, 6227, 6228, 6229, 6230, 6231, 6232, 6233, 6234, 6235, 6236, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 6244, 6245, 6246, 6247, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255, 6256, 6257, 6258, 6259, 6260, 6261, 6262, 6263, 6264, 6265, 6266, 6267, 6268, 6269, 6270, 6271, 6272, 6273, 6274, 6275, 6276, 6277, 6278, 6279, 6280, 6281, 6282, 6283, 6284, 6285, 6286, 6287, 6288, 6289, 6290, 6291, 6292, 6293, 6294, 6295, 6296, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6321, 6322, 6323, 6324, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6339, 6340, 6341, 6342, 6343, 6344, 6345, 6346, 6347, 6348, 6349, 6350, 6351, 6352, 6353, 6354, 6355, 6356, 6357, 6358, 6359, 6360, 6361, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6369, 6370, 6371, 6372, 6373, 6374, 6375, 6376, 6377, 6378, 6379, 6380, 6381, 6382, 6383, 6384, 6385, 6386, 6387, 6388, 6389, 6390, 6391, 6392, 6393, 6394, 6395, 6396, 6397, 6398, 6399, 6400, 6401, 6402, 6403, 6404, 6405, 6406, 6407, 6408, 6409, 6410, 6411, 6412, 6413, 6414, 6415, 6416, 6417, 6418, 6419, 6420, 6421, 6422, 6423, 6424, 6425, 6426, 6427, 6428, 6429, 6430, 6431, 6432, 6433, 6434, 6435, 6436, 6437, 6438, 6439, 6440, 6441, 6442, 6443, 6444, 6445, 6446, 6447, 6448, 6449, 6450, 6451, 6452, 6453, 6454, 6455, 6456, 6457, 6458, 6459, 6460, 6461, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6469, 6470, 6471, 6472, 6473, 6474, 6475, 6476, 6477, 6478, 6479, 6480, 6481, 6482, 6483, 6484, 6485, 6486, 6487, 6488, 6489, 6490, 6491, 6492, 6493, 6494, 6495, 6496, 6497, 6498, 6499, 6500, 6501, 6502, 6503, 6504, 6505, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6517, 6518, 6519, 6520, 6521, 6522, 6523, 6524, 6525, 6526, 6527, 6528, 6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6537, 6538, 6539, 6540, 6541, 6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550, 6551, 6552, 6553, 6554, 6555, 6556, 6557, 6558, 6559, 6560, 6561, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6569, 6570, 6571, 6572, 6573, 6574, 6575, 6576, 6577, 6578, 6579, 6580, 6581, 6582, 6583, 6584, 6585, 6586, 6587, 6588, 6589, 6590, 6591, 6592, 6593, 6594, 6595, 6596, 6597, 6598, 6599, 6600, 6601, 6602, 6603, 6604, 6605, 6606, 6607, 6608, 6609, 6610, 6611, 6612, 6613, 6614, 6615, 6616, 6617, 6618, 6619, 6620, 6621, 6622, 6623, 6624, 6625, 6626, 6627, 6628, 6629, 6630, 6631, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6639, 6640, 6641, 6642, 6643, 6644, 6645, 6646, 6647, 6648, 6649, 6650, 6651, 6652, 6653, 6654, 6655, 6656, 6657, 6658, 6659, 6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679, 6680, 6681, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692, 6693, 6694, 6695, 6696, 6697, 6698, 6699, 6700, 6701, 6702, 6703, 6704, 6705, 6706, 6707, 6708, 6709, 6710, 6711, 6712, 6713, 6714, 6715, 6716, 6717, 6718, 6719, 6720, 6721, 6722, 6723, 6724, 6725, 6726, 6727, 6728, 6729, 6730, 6731, 6732, 6733, 6734, 6735, 6736, 6737, 6738, 6739, 6740, 6741, 6742, 6743, 6744, 6745, 6746, 6747, 6748, 6749, 6750, 6751, 6752, 6753, 6754, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775, 6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6793, 6794, 6795, 6796, 6797, 6798, 6799, 6800, 6801, 6802, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810, 6811, 6812, 6813, 6814, 6815, 6816, 6817, 6818, 6819, 6820, 6821, 6822, 6823, 6824, 6825, 6826, 6827, 6828, 6829, 6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838, 6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6867, 6868, 6869, 6870, 6871, 6872, 6873, 6874, 6875, 6876, 6877, 6878, 6879, 6880, 6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 6907, 6908, 6909, 6910, 6911, 6912
# .float 6913, 6914, 6915, 6916, 6917, 6918, 6919, 6920, 6921, 6922, 6923, 6924, 6925, 6926, 6927, 6928, 6929, 6930, 6931, 6932, 6933, 6934, 6935, 6936, 6937, 6938, 6939, 6940, 6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948, 6949, 6950, 6951, 6952, 6953, 6954, 6955, 6956, 6957, 6958, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971, 6972, 6973, 6974, 6975, 6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984, 6985, 6986, 6987, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996, 6997, 6998, 6999, 7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035, 7036, 7037, 7038, 7039, 7040, 7041, 7042, 7043, 7044, 7045, 7046, 7047, 7048, 7049, 7050, 7051, 7052, 7053, 7054, 7055, 7056, 7057, 7058, 7059, 7060, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088, 7089, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7163, 7164, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 7185, 7186, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7289, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7321, 7322, 7323, 7324, 7325, 7326, 7327, 7328, 7329, 7330, 7331, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7341, 7342, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7351, 7352, 7353, 7354, 7355, 7356, 7357, 7358, 7359, 7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369, 7370, 7371, 7372, 7373, 7374, 7375, 7376, 7377, 7378, 7379, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7432, 7433, 7434, 7435, 7436, 7437, 7438, 7439, 7440, 7441, 7442, 7443, 7444, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454, 7455, 7456, 7457, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484, 7485, 7486, 7487, 7488, 7489, 7490, 7491, 7492, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7517, 7518, 7519, 7520, 7521, 7522, 7523, 7524, 7525, 7526, 7527, 7528, 7529, 7530, 7531, 7532, 7533, 7534, 7535, 7536, 7537, 7538, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547, 7548, 7549, 7550, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560, 7561, 7562, 7563, 7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573, 7574, 7575, 7576, 7577, 7578, 7579, 7580, 7581, 7582, 7583, 7584, 7585, 7586, 7587, 7588, 7589, 7590, 7591, 7592, 7593, 7594, 7595, 7596, 7597, 7598, 7599, 7600, 7601, 7602, 7603, 7604, 7605, 7606, 7607, 7608, 7609, 7610, 7611, 7612, 7613, 7614, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7624, 7625, 7626, 7627, 7628, 7629, 7630, 7631, 7632, 7633, 7634, 7635, 7636, 7637, 7638, 7639, 7640, 7641, 7642, 7643, 7644, 7645, 7646, 7647, 7648, 7649, 7650, 7651, 7652, 7653, 7654, 7655, 7656, 7657, 7658, 7659, 7660, 7661, 7662, 7663, 7664, 7665, 7666, 7667, 7668, 7669, 7670, 7671, 7672, 7673, 7674, 7675, 7676, 7677, 7678, 7679, 7680, 7681, 7682, 7683, 7684, 7685, 7686, 7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694, 7695, 7696, 7697, 7698, 7699, 7700, 7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7718, 7719, 7720, 7721, 7722, 7723, 7724, 7725, 7726, 7727, 7728, 7729, 7730, 7731, 7732, 7733, 7734, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7748, 7749, 7750, 7751, 7752, 7753, 7754, 7755, 7756, 7757, 7758, 7759, 7760, 7761, 7762, 7763, 7764, 7765, 7766, 7767, 7768, 7769, 7770, 7771, 7772, 7773, 7774, 7775, 7776, 7777, 7778, 7779, 7780, 7781, 7782, 7783, 7784, 7785, 7786, 7787, 7788, 7789, 7790, 7791, 7792, 7793, 7794, 7795, 7796, 7797, 7798, 7799, 7800, 7801, 7802, 7803, 7804, 7805, 7806, 7807, 7808, 7809, 7810, 7811, 7812, 7813, 7814, 7815, 7816, 7817, 7818, 7819, 7820, 7821, 7822, 7823, 7824, 7825, 7826, 7827, 7828, 7829, 7830, 7831, 7832, 7833, 7834, 7835, 7836, 7837, 7838, 7839, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883, 7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7896, 7897, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7907, 7908, 7909, 7910, 7911, 7912, 7913, 7914, 7915, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924, 7925, 7926, 7927, 7928, 7929, 7930, 7931, 7932, 7933, 7934, 7935, 7936, 7937, 7938, 7939, 7940, 7941, 7942, 7943, 7944, 7945, 7946, 7947, 7948, 7949, 7950, 7951, 7952, 7953, 7954, 7955, 7956, 7957, 7958, 7959, 7960, 7961, 7962, 7963, 7964, 7965, 7966, 7967, 7968, 7969, 7970, 7971, 7972, 7973, 7974, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 7988, 7989, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 7997, 7998, 7999, 8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015, 8016, 8017, 8018, 8019, 8020, 8021, 8022, 8023, 8024, 8025, 8026, 8027, 8028, 8029, 8030, 8031, 8032, 8033, 8034, 8035, 8036, 8037, 8038, 8039, 8040, 8041, 8042, 8043, 8044, 8045, 8046, 8047, 8048, 8049, 8050, 8051, 8052, 8053, 8054, 8055, 8056, 8057, 8058, 8059, 8060, 8061, 8062, 8063, 8064
# .float 8065, 8066, 8067, 8068, 8069, 8070, 8071, 8072, 8073, 8074, 8075, 8076, 8077, 8078, 8079, 8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089, 8090, 8091, 8092, 8093, 8094, 8095, 8096, 8097, 8098, 8099, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112, 8113, 8114, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122, 8123, 8124, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138, 8139, 8140, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8164, 8165, 8166, 8167, 8168, 8169, 8170, 8171, 8172, 8173, 8174, 8175, 8176, 8177, 8178, 8179, 8180, 8181, 8182, 8183, 8184, 8185, 8186, 8187, 8188, 8189, 8190, 8191, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8225, 8226, 8227, 8228, 8229, 8230, 8231, 8232, 8233, 8234, 8235, 8236, 8237, 8238, 8239, 8240, 8241, 8242, 8243, 8244, 8245, 8246, 8247, 8248, 8249, 8250, 8251, 8252, 8253, 8254, 8255, 8256, 8257, 8258, 8259, 8260, 8261, 8262, 8263, 8264, 8265, 8266, 8267, 8268, 8269, 8270, 8271, 8272, 8273, 8274, 8275, 8276, 8277, 8278, 8279, 8280, 8281, 8282, 8283, 8284, 8285, 8286, 8287, 8288, 8289, 8290, 8291, 8292, 8293, 8294, 8295, 8296, 8297, 8298, 8299, 8300, 8301, 8302, 8303, 8304, 8305, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315, 8316, 8317, 8318, 8319, 8320, 8321, 8322, 8323, 8324, 8325, 8326, 8327, 8328, 8329, 8330, 8331, 8332, 8333, 8334, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8347, 8348, 8349, 8350, 8351, 8352, 8353, 8354, 8355, 8356, 8357, 8358, 8359, 8360, 8361, 8362, 8363, 8364, 8365, 8366, 8367, 8368, 8369, 8370, 8371, 8372, 8373, 8374, 8375, 8376, 8377, 8378, 8379, 8380, 8381, 8382, 8383, 8384, 8385, 8386, 8387, 8388, 8389, 8390, 8391, 8392, 8393, 8394, 8395, 8396, 8397, 8398, 8399, 8400, 8401, 8402, 8403, 8404, 8405, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424, 8425, 8426, 8427, 8428, 8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436, 8437, 8438, 8439, 8440, 8441, 8442, 8443, 8444, 8445, 8446, 8447, 8448, 8449, 8450, 8451, 8452, 8453, 8454, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8477, 8478, 8479, 8480, 8481, 8482, 8483, 8484, 8485, 8486, 8487, 8488, 8489, 8490, 8491, 8492, 8493, 8494, 8495, 8496, 8497, 8498, 8499, 8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511, 8512, 8513, 8514, 8515, 8516, 8517, 8518, 8519, 8520, 8521, 8522, 8523, 8524, 8525, 8526, 8527, 8528, 8529, 8530, 8531, 8532, 8533, 8534, 8535, 8536, 8537, 8538, 8539, 8540, 8541, 8542, 8543, 8544, 8545, 8546, 8547, 8548, 8549, 8550, 8551, 8552, 8553, 8554, 8555, 8556, 8557, 8558, 8559, 8560, 8561, 8562, 8563, 8564, 8565, 8566, 8567, 8568, 8569, 8570, 8571, 8572, 8573, 8574, 8575, 8576, 8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8588, 8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8597, 8598, 8599, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8611, 8612, 8613, 8614, 8615, 8616, 8617, 8618, 8619, 8620, 8621, 8622, 8623, 8624, 8625, 8626, 8627, 8628, 8629, 8630, 8631, 8632, 8633, 8634, 8635, 8636, 8637, 8638, 8639, 8640, 8641, 8642, 8643, 8644, 8645, 8646, 8647, 8648, 8649, 8650, 8651, 8652, 8653, 8654, 8655, 8656, 8657, 8658, 8659, 8660, 8661, 8662, 8663, 8664, 8665, 8666, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674, 8675, 8676, 8677, 8678, 8679, 8680, 8681, 8682, 8683, 8684, 8685, 8686, 8687, 8688, 8689, 8690, 8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8717, 8718, 8719, 8720, 8721, 8722, 8723, 8724, 8725, 8726, 8727, 8728, 8729, 8730, 8731, 8732, 8733, 8734, 8735, 8736, 8737, 8738, 8739, 8740, 8741, 8742, 8743, 8744, 8745, 8746, 8747, 8748, 8749, 8750, 8751, 8752, 8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764, 8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 8777, 8778, 8779, 8780, 8781, 8782, 8783, 8784, 8785, 8786, 8787, 8788, 8789, 8790, 8791, 8792, 8793, 8794, 8795, 8796, 8797, 8798, 8799, 8800, 8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808, 8809, 8810, 8811, 8812, 8813, 8814, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8828, 8829, 8830, 8831, 8832, 8833, 8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8850, 8851, 8852, 8853, 8854, 8855, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8871, 8872, 8873, 8874, 8875, 8876, 8877, 8878, 8879, 8880, 8881, 8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8890, 8891, 8892, 8893, 8894, 8895, 8896, 8897, 8898, 8899, 8900, 8901, 8902, 8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8944, 8945, 8946, 8947, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8965, 8966, 8967, 8968, 8969, 8970, 8971, 8972, 8973, 8974, 8975, 8976, 8977, 8978, 8979, 8980, 8981, 8982, 8983, 8984, 8985, 8986, 8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994, 8995, 8996, 8997, 8998, 8999, 9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 9012, 9013, 9014, 9015, 9016, 9017, 9018, 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027, 9028, 9029, 9030, 9031, 9032, 9033, 9034, 9035, 9036, 9037, 9038, 9039, 9040, 9041, 9042, 9043, 9044, 9045, 9046, 9047, 9048, 9049, 9050, 9051, 9052, 9053, 9054, 9055, 9056, 9057, 9058, 9059, 9060, 9061, 9062, 9063, 9064, 9065, 9066, 9067, 9068, 9069, 9070, 9071, 9072, 9073, 9074, 9075, 9076, 9077, 9078, 9079, 9080, 9081, 9082, 9083, 9084, 9085, 9086, 9087, 9088, 9089, 9090, 9091, 9092, 9093, 9094, 9095, 9096, 9097, 9098, 9099, 9100, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110, 9111, 9112, 9113, 9114, 9115, 9116, 9117, 9118, 9119, 9120, 9121, 9122, 9123, 9124, 9125, 9126, 9127, 9128, 9129, 9130, 9131, 9132, 9133, 9134, 9135, 9136, 9137, 9138, 9139, 9140, 9141, 9142, 9143, 9144, 9145, 9146, 9147, 9148, 9149, 9150, 9151, 9152, 9153, 9154, 9155, 9156, 9157, 9158, 9159, 9160, 9161, 9162, 9163, 9164, 9165, 9166, 9167, 9168, 9169, 9170, 9171, 9172, 9173, 9174, 9175, 9176, 9177, 9178, 9179, 9180, 9181, 9182, 9183, 9184, 9185, 9186, 9187, 9188, 9189, 9190, 9191, 9192, 9193, 9194, 9195, 9196, 9197, 9198, 9199, 9200, 9201, 9202, 9203, 9204, 9205, 9206, 9207, 9208, 9209, 9210, 9211, 9212, 9213, 9214, 9215, 9216
# .float 9217, 9218, 9219, 9220, 9221, 9222, 9223, 9224, 9225, 9226, 9227, 9228, 9229, 9230, 9231, 9232, 9233, 9234, 9235, 9236, 9237, 9238, 9239, 9240, 9241, 9242, 9243, 9244, 9245, 9246, 9247, 9248, 9249, 9250, 9251, 9252, 9253, 9254, 9255, 9256, 9257, 9258, 9259, 9260, 9261, 9262, 9263, 9264, 9265, 9266, 9267, 9268, 9269, 9270, 9271, 9272, 9273, 9274, 9275, 9276, 9277, 9278, 9279, 9280, 9281, 9282, 9283, 9284, 9285, 9286, 9287, 9288, 9289, 9290, 9291, 9292, 9293, 9294, 9295, 9296, 9297, 9298, 9299, 9300, 9301, 9302, 9303, 9304, 9305, 9306, 9307, 9308, 9309, 9310, 9311, 9312, 9313, 9314, 9315, 9316, 9317, 9318, 9319, 9320, 9321, 9322, 9323, 9324, 9325, 9326, 9327, 9328, 9329, 9330, 9331, 9332, 9333, 9334, 9335, 9336, 9337, 9338, 9339, 9340, 9341, 9342, 9343, 9344, 9345, 9346, 9347, 9348, 9349, 9350, 9351, 9352, 9353, 9354, 9355, 9356, 9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364, 9365, 9366, 9367, 9368, 9369, 9370, 9371, 9372, 9373, 9374, 9375, 9376, 9377, 9378, 9379, 9380, 9381, 9382, 9383, 9384, 9385, 9386, 9387, 9388, 9389, 9390, 9391, 9392, 9393, 9394, 9395, 9396, 9397, 9398, 9399, 9400, 9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408, 9409, 9410, 9411, 9412, 9413, 9414, 9415, 9416, 9417, 9418, 9419, 9420, 9421, 9422, 9423, 9424, 9425, 9426, 9427, 9428, 9429, 9430, 9431, 9432, 9433, 9434, 9435, 9436, 9437, 9438, 9439, 9440, 9441, 9442, 9443, 9444, 9445, 9446, 9447, 9448, 9449, 9450, 9451, 9452, 9453, 9454, 9455, 9456, 9457, 9458, 9459, 9460, 9461, 9462, 9463, 9464, 9465, 9466, 9467, 9468, 9469, 9470, 9471, 9472, 9473, 9474, 9475, 9476, 9477, 9478, 9479, 9480, 9481, 9482, 9483, 9484, 9485, 9486, 9487, 9488, 9489, 9490, 9491, 9492, 9493, 9494, 9495, 9496, 9497, 9498, 9499, 9500, 9501, 9502, 9503, 9504, 9505, 9506, 9507, 9508, 9509, 9510, 9511, 9512, 9513, 9514, 9515, 9516, 9517, 9518, 9519, 9520, 9521, 9522, 9523, 9524, 9525, 9526, 9527, 9528, 9529, 9530, 9531, 9532, 9533, 9534, 9535, 9536, 9537, 9538, 9539, 9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9548, 9549, 9550, 9551, 9552, 9553, 9554, 9555, 9556, 9557, 9558, 9559, 9560, 9561, 9562, 9563, 9564, 9565, 9566, 9567, 9568, 9569, 9570, 9571, 9572, 9573, 9574, 9575, 9576, 9577, 9578, 9579, 9580, 9581, 9582, 9583, 9584, 9585, 9586, 9587, 9588, 9589, 9590, 9591, 9592, 9593, 9594, 9595, 9596, 9597, 9598, 9599, 9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9610, 9611, 9612, 9613, 9614, 9615, 9616, 9617, 9618, 9619, 9620, 9621, 9622, 9623, 9624, 9625, 9626, 9627, 9628, 9629, 9630, 9631, 9632, 9633, 9634, 9635, 9636, 9637, 9638, 9639, 9640, 9641, 9642, 9643, 9644, 9645, 9646, 9647, 9648, 9649, 9650, 9651, 9652, 9653, 9654, 9655, 9656, 9657, 9658, 9659, 9660, 9661, 9662, 9663, 9664, 9665, 9666, 9667, 9668, 9669, 9670, 9671, 9672, 9673, 9674, 9675, 9676, 9677, 9678, 9679, 9680, 9681, 9682, 9683, 9684, 9685, 9686, 9687, 9688, 9689, 9690, 9691, 9692, 9693, 9694, 9695, 9696, 9697, 9698, 9699, 9700, 9701, 9702, 9703, 9704, 9705, 9706, 9707, 9708, 9709, 9710, 9711, 9712, 9713, 9714, 9715, 9716, 9717, 9718, 9719, 9720, 9721, 9722, 9723, 9724, 9725, 9726, 9727, 9728, 9729, 9730, 9731, 9732, 9733, 9734, 9735, 9736, 9737, 9738, 9739, 9740, 9741, 9742, 9743, 9744, 9745, 9746, 9747, 9748, 9749, 9750, 9751, 9752, 9753, 9754, 9755, 9756, 9757, 9758, 9759, 9760, 9761, 9762, 9763, 9764, 9765, 9766, 9767, 9768, 9769, 9770, 9771, 9772, 9773, 9774, 9775, 9776, 9777, 9778, 9779, 9780, 9781, 9782, 9783, 9784, 9785, 9786, 9787, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9798, 9799, 9800, 9801, 9802, 9803, 9804, 9805, 9806, 9807, 9808, 9809, 9810, 9811, 9812, 9813, 9814, 9815, 9816, 9817, 9818, 9819, 9820, 9821, 9822, 9823, 9824, 9825, 9826, 9827, 9828, 9829, 9830, 9831, 9832, 9833, 9834, 9835, 9836, 9837, 9838, 9839, 9840, 9841, 9842, 9843, 9844, 9845, 9846, 9847, 9848, 9849, 9850, 9851, 9852, 9853, 9854, 9855, 9856, 9857, 9858, 9859, 9860, 9861, 9862, 9863, 9864, 9865, 9866, 9867, 9868, 9869, 9870, 9871, 9872, 9873, 9874, 9875, 9876, 9877, 9878, 9879, 9880, 9881, 9882, 9883, 9884, 9885, 9886, 9887, 9888, 9889, 9890, 9891, 9892, 9893, 9894, 9895, 9896, 9897, 9898, 9899, 9900, 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908, 9909, 9910, 9911, 9912, 9913, 9914, 9915, 9916, 9917, 9918, 9919, 9920, 9921, 9922, 9923, 9924, 9925, 9926, 9927, 9928, 9929, 9930, 9931, 9932, 9933, 9934, 9935, 9936, 9937, 9938, 9939, 9940, 9941, 9942, 9943, 9944, 9945, 9946, 9947, 9948, 9949, 9950, 9951, 9952, 9953, 9954, 9955, 9956, 9957, 9958, 9959, 9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967, 9968, 9969, 9970, 9971, 9972, 9973, 9974, 9975, 9976, 9977, 9978, 9979, 9980, 9981, 9982, 9983, 9984, 9985, 9986, 9987, 9988, 9989, 9990, 9991, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030, 10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10042, 10043, 10044, 10045, 10046, 10047, 10048, 10049, 10050, 10051, 10052, 10053, 10054, 10055, 10056, 10057, 10058, 10059, 10060, 10061, 10062, 10063, 10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073, 10074, 10075, 10076, 10077, 10078, 10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089, 10090, 10091, 10092, 10093, 10094, 10095, 10096, 10097, 10098, 10099, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10107, 10108, 10109, 10110, 10111, 10112, 10113, 10114, 10115, 10116, 10117, 10118, 10119, 10120, 10121, 10122, 10123, 10124, 10125, 10126, 10127, 10128, 10129, 10130, 10131, 10132, 10133, 10134, 10135, 10136, 10137, 10138, 10139, 10140, 10141, 10142, 10143, 10144, 10145, 10146, 10147, 10148, 10149, 10150, 10151, 10152, 10153, 10154, 10155, 10156, 10157, 10158, 10159, 10160, 10161, 10162, 10163, 10164, 10165, 10166, 10167, 10168, 10169, 10170, 10171, 10172, 10173, 10174, 10175, 10176, 10177, 10178, 10179, 10180, 10181, 10182, 10183, 10184, 10185, 10186, 10187, 10188, 10189, 10190, 10191, 10192, 10193, 10194, 10195, 10196, 10197, 10198, 10199, 10200, 10201, 10202, 10203, 10204, 10205, 10206, 10207, 10208, 10209, 10210, 10211, 10212, 10213, 10214, 10215, 10216, 10217, 10218, 10219, 10220, 10221, 10222, 10223, 10224, 10225, 10226, 10227, 10228, 10229, 10230, 10231, 10232, 10233, 10234, 10235, 10236, 10237, 10238, 10239, 10240, 10241, 10242, 10243, 10244, 10245, 10246, 10247, 10248, 10249, 10250, 10251, 10252, 10253, 10254, 10255, 10256, 10257, 10258, 10259, 10260, 10261, 10262, 10263, 10264, 10265, 10266, 10267, 10268, 10269, 10270, 10271, 10272, 10273, 10274, 10275, 10276, 10277, 10278, 10279, 10280, 10281, 10282, 10283, 10284, 10285, 10286, 10287, 10288, 10289, 10290, 10291, 10292, 10293, 10294, 10295, 10296, 10297, 10298, 10299, 10300, 10301, 10302, 10303, 10304, 10305, 10306, 10307, 10308, 10309, 10310, 10311, 10312, 10313, 10314, 10315, 10316, 10317, 10318, 10319, 10320, 10321, 10322, 10323, 10324, 10325, 10326, 10327, 10328, 10329, 10330, 10331, 10332, 10333, 10334, 10335, 10336, 10337, 10338, 10339, 10340, 10341, 10342, 10343, 10344, 10345, 10346, 10347, 10348, 10349, 10350, 10351, 10352, 10353, 10354, 10355, 10356, 10357, 10358, 10359, 10360, 10361, 10362, 10363, 10364, 10365, 10366, 10367, 10368
# .float 10369, 10370, 10371, 10372, 10373, 10374, 10375, 10376, 10377, 10378, 10379, 10380, 10381, 10382, 10383, 10384, 10385, 10386, 10387, 10388, 10389, 10390, 10391, 10392, 10393, 10394, 10395, 10396, 10397, 10398, 10399, 10400, 10401, 10402, 10403, 10404, 10405, 10406, 10407, 10408, 10409, 10410, 10411, 10412, 10413, 10414, 10415, 10416, 10417, 10418, 10419, 10420, 10421, 10422, 10423, 10424, 10425, 10426, 10427, 10428, 10429, 10430, 10431, 10432, 10433, 10434, 10435, 10436, 10437, 10438, 10439, 10440, 10441, 10442, 10443, 10444, 10445, 10446, 10447, 10448, 10449, 10450, 10451, 10452, 10453, 10454, 10455, 10456, 10457, 10458, 10459, 10460, 10461, 10462, 10463, 10464, 10465, 10466, 10467, 10468, 10469, 10470, 10471, 10472, 10473, 10474, 10475, 10476, 10477, 10478, 10479, 10480, 10481, 10482, 10483, 10484, 10485, 10486, 10487, 10488, 10489, 10490, 10491, 10492, 10493, 10494, 10495, 10496, 10497, 10498, 10499, 10500, 10501, 10502, 10503, 10504, 10505, 10506, 10507, 10508, 10509, 10510, 10511, 10512, 10513, 10514, 10515, 10516, 10517, 10518, 10519, 10520, 10521, 10522, 10523, 10524, 10525, 10526, 10527, 10528, 10529, 10530, 10531, 10532, 10533, 10534, 10535, 10536, 10537, 10538, 10539, 10540, 10541, 10542, 10543, 10544, 10545, 10546, 10547, 10548, 10549, 10550, 10551, 10552, 10553, 10554, 10555, 10556, 10557, 10558, 10559, 10560, 10561, 10562, 10563, 10564, 10565, 10566, 10567, 10568, 10569, 10570, 10571, 10572, 10573, 10574, 10575, 10576, 10577, 10578, 10579, 10580, 10581, 10582, 10583, 10584, 10585, 10586, 10587, 10588, 10589, 10590, 10591, 10592, 10593, 10594, 10595, 10596, 10597, 10598, 10599, 10600, 10601, 10602, 10603, 10604, 10605, 10606, 10607, 10608, 10609, 10610, 10611, 10612, 10613, 10614, 10615, 10616, 10617, 10618, 10619, 10620, 10621, 10622, 10623, 10624, 10625, 10626, 10627, 10628, 10629, 10630, 10631, 10632, 10633, 10634, 10635, 10636, 10637, 10638, 10639, 10640, 10641, 10642, 10643, 10644, 10645, 10646, 10647, 10648, 10649, 10650, 10651, 10652, 10653, 10654, 10655, 10656, 10657, 10658, 10659, 10660, 10661, 10662, 10663, 10664, 10665, 10666, 10667, 10668, 10669, 10670, 10671, 10672, 10673, 10674, 10675, 10676, 10677, 10678, 10679, 10680, 10681, 10682, 10683, 10684, 10685, 10686, 10687, 10688, 10689, 10690, 10691, 10692, 10693, 10694, 10695, 10696, 10697, 10698, 10699, 10700, 10701, 10702, 10703, 10704, 10705, 10706, 10707, 10708, 10709, 10710, 10711, 10712, 10713, 10714, 10715, 10716, 10717, 10718, 10719, 10720, 10721, 10722, 10723, 10724, 10725, 10726, 10727, 10728, 10729, 10730, 10731, 10732, 10733, 10734, 10735, 10736, 10737, 10738, 10739, 10740, 10741, 10742, 10743, 10744, 10745, 10746, 10747, 10748, 10749, 10750, 10751, 10752, 10753, 10754, 10755, 10756, 10757, 10758, 10759, 10760, 10761, 10762, 10763, 10764, 10765, 10766, 10767, 10768, 10769, 10770, 10771, 10772, 10773, 10774, 10775, 10776, 10777, 10778, 10779, 10780, 10781, 10782, 10783, 10784, 10785, 10786, 10787, 10788, 10789, 10790, 10791, 10792, 10793, 10794, 10795, 10796, 10797, 10798, 10799, 10800, 10801, 10802, 10803, 10804, 10805, 10806, 10807, 10808, 10809, 10810, 10811, 10812, 10813, 10814, 10815, 10816, 10817, 10818, 10819, 10820, 10821, 10822, 10823, 10824, 10825, 10826, 10827, 10828, 10829, 10830, 10831, 10832, 10833, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 10842, 10843, 10844, 10845, 10846, 10847, 10848, 10849, 10850, 10851, 10852, 10853, 10854, 10855, 10856, 10857, 10858, 10859, 10860, 10861, 10862, 10863, 10864, 10865, 10866, 10867, 10868, 10869, 10870, 10871, 10872, 10873, 10874, 10875, 10876, 10877, 10878, 10879, 10880, 10881, 10882, 10883, 10884, 10885, 10886, 10887, 10888, 10889, 10890, 10891, 10892, 10893, 10894, 10895, 10896, 10897, 10898, 10899, 10900, 10901, 10902, 10903, 10904, 10905, 10906, 10907, 10908, 10909, 10910, 10911, 10912, 10913, 10914, 10915, 10916, 10917, 10918, 10919, 10920, 10921, 10922, 10923, 10924, 10925, 10926, 10927, 10928, 10929, 10930, 10931, 10932, 10933, 10934, 10935, 10936, 10937, 10938, 10939, 10940, 10941, 10942, 10943, 10944, 10945, 10946, 10947, 10948, 10949, 10950, 10951, 10952, 10953, 10954, 10955, 10956, 10957, 10958, 10959, 10960, 10961, 10962, 10963, 10964, 10965, 10966, 10967, 10968, 10969, 10970, 10971, 10972, 10973, 10974, 10975, 10976, 10977, 10978, 10979, 10980, 10981, 10982, 10983, 10984, 10985, 10986, 10987, 10988, 10989, 10990, 10991, 10992, 10993, 10994, 10995, 10996, 10997, 10998, 10999, 11000, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014, 11015, 11016, 11017, 11018, 11019, 11020, 11021, 11022, 11023, 11024, 11025, 11026, 11027, 11028, 11029, 11030, 11031, 11032, 11033, 11034, 11035, 11036, 11037, 11038, 11039, 11040, 11041, 11042, 11043, 11044, 11045, 11046, 11047, 11048, 11049, 11050, 11051, 11052, 11053, 11054, 11055, 11056, 11057, 11058, 11059, 11060, 11061, 11062, 11063, 11064, 11065, 11066, 11067, 11068, 11069, 11070, 11071, 11072, 11073, 11074, 11075, 11076, 11077, 11078, 11079, 11080, 11081, 11082, 11083, 11084, 11085, 11086, 11087, 11088, 11089, 11090, 11091, 11092, 11093, 11094, 11095, 11096, 11097, 11098, 11099, 11100, 11101, 11102, 11103, 11104, 11105, 11106, 11107, 11108, 11109, 11110, 11111, 11112, 11113, 11114, 11115, 11116, 11117, 11118, 11119, 11120, 11121, 11122, 11123, 11124, 11125, 11126, 11127, 11128, 11129, 11130, 11131, 11132, 11133, 11134, 11135, 11136, 11137, 11138, 11139, 11140, 11141, 11142, 11143, 11144, 11145, 11146, 11147, 11148, 11149, 11150, 11151, 11152, 11153, 11154, 11155, 11156, 11157, 11158, 11159, 11160, 11161, 11162, 11163, 11164, 11165, 11166, 11167, 11168, 11169, 11170, 11171, 11172, 11173, 11174, 11175, 11176, 11177, 11178, 11179, 11180, 11181, 11182, 11183, 11184, 11185, 11186, 11187, 11188, 11189, 11190, 11191, 11192, 11193, 11194, 11195, 11196, 11197, 11198, 11199, 11200, 11201, 11202, 11203, 11204, 11205, 11206, 11207, 11208, 11209, 11210, 11211, 11212, 11213, 11214, 11215, 11216, 11217, 11218, 11219, 11220, 11221, 11222, 11223, 11224, 11225, 11226, 11227, 11228, 11229, 11230, 11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243, 11244, 11245, 11246, 11247, 11248, 11249, 11250, 11251, 11252, 11253, 11254, 11255, 11256, 11257, 11258, 11259, 11260, 11261, 11262, 11263, 11264, 11265, 11266, 11267, 11268, 11269, 11270, 11271, 11272, 11273, 11274, 11275, 11276, 11277, 11278, 11279, 11280, 11281, 11282, 11283, 11284, 11285, 11286, 11287, 11288, 11289, 11290, 11291, 11292, 11293, 11294, 11295, 11296, 11297, 11298, 11299, 11300, 11301, 11302, 11303, 11304, 11305, 11306, 11307, 11308, 11309, 11310, 11311, 11312, 11313, 11314, 11315, 11316, 11317, 11318, 11319, 11320, 11321, 11322, 11323, 11324, 11325, 11326, 11327, 11328, 11329, 11330, 11331, 11332, 11333, 11334, 11335, 11336, 11337, 11338, 11339, 11340, 11341, 11342, 11343, 11344, 11345, 11346, 11347, 11348, 11349, 11350, 11351, 11352, 11353, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11361, 11362, 11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11371, 11372, 11373, 11374, 11375, 11376, 11377, 11378, 11379, 11380, 11381, 11382, 11383, 11384, 11385, 11386, 11387, 11388, 11389, 11390, 11391, 11392, 11393, 11394, 11395, 11396, 11397, 11398, 11399, 11400, 11401, 11402, 11403, 11404, 11405, 11406, 11407, 11408, 11409, 11410, 11411, 11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 11424, 11425, 11426, 11427, 11428, 11429, 11430, 11431, 11432, 11433, 11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441, 11442, 11443, 11444, 11445, 11446, 11447, 11448, 11449, 11450, 11451, 11452, 11453, 11454, 11455, 11456, 11457, 11458, 11459, 11460, 11461, 11462, 11463, 11464, 11465, 11466, 11467, 11468, 11469, 11470, 11471, 11472, 11473, 11474, 11475, 11476, 11477, 11478, 11479, 11480, 11481, 11482, 11483, 11484, 11485, 11486, 11487, 11488, 11489, 11490, 11491, 11492, 11493, 11494, 11495, 11496, 11497, 11498, 11499, 11500, 11501, 11502, 11503, 11504, 11505, 11506, 11507, 11508, 11509, 11510, 11511, 11512, 11513, 11514, 11515, 11516, 11517, 11518, 11519, 11520

# b:    .float 1,2,3,4,5,6,7,8,9,10                # b[10]

# y:    .float 0, 0, 0, 0, 0, 0, 0, 0, 0, 0                     # y[10]

# .text
# .globl _start
# _start:
#     # Initialize base addresses
#     la a0, x              # a0 = &x[0] (pointer to x array)
#     la a1, W              # a1 = &W[0][0] (pointer to W matrix)
#     la a2, b              # a2 = &b[0] (pointer to b array)
#     la a3, y              # a3 = &y[0] (pointer to y array)

#     li t0, 0              # i = 0 (output index)
#     li t1, 10             # OUTPUT_DIM (10)
#     li t4, 1152           # INPUT_DIM (1152)

# outer_loop:
#     li t2, 0              # j = 0 (input index)
#     fmv.s.x f0, x0

# inner_loop:
#     bge t2, t4, inner_loop_end  # if j >= INPUT_DIM, exit inner loop

#     # Load W[i][j] as a floating-point value
#     mul t5, t0, t4        # i * INPUT_DIM
#     add t5, t5, t2        # index = i * INPUT_DIM + j
#     slli t5, t5, 2        # offset in bytes (multiply by 4)
#     add t6, a1, t5        # addr of W[i][j]
#     flw f1, 0(t6)         # f1 = W[i][j] (load W[i][j] as float)

#     # Load x[j] as a floating-point value
#     slli t5, t2, 2        # offset to x[j]
#     add t6, a0, t5
#     flw f2, 0(t6)         # f2 = x[j] (load x[j] as float)

#     # Multiply and add in floating-point
#     fmul.s f3, f1, f2     # f3 = W[i][j] * x[j] (floating-point multiplication)
#     fadd.s f0, f0, f3     # sum += W[i][j] * x[j] (floating-point addition)

#     # j++
#     addi t2, t2, 1
#     blt t2, t4, inner_loop

# inner_loop_end:
#     # Add bias (floating-point addition)
#     slli t5, t0, 2        # offset to b[i]
#     add t6, a2, t5
#     flw f1, 0(t6)         # f1 = b[i] (load b[i] as float)
#     fadd.s f0, f0, f1     # sum += b[i] (floating-point addition)

#     # Store y[i] (floating-point value)
#     slli t5, t0, 2        # offset to y[i]
#     add t6, a3, t5
#     fsw f0, 0(t6)         # store sum into y[i] as float

#     # i++
#     addi t0, t0, 1
#     li t1, 10             # OUTPUT_DIM (10)
#     blt t0, t1, outer_loop

#     # Keep this part for testing:
#     # Load y values into f4 for testing (should be flw instead of fsw)
#     flw f4, 0(a3)         # Load the first y value into f4

#     # Load y values into f5 for testing (should be flw instead of fsw)
#     flw f5, 4(a3)         # Load the second y value into f5

#     # Done (endless loop or ecall to exit)
#     ecall
# done:
#     j done



# Softmax Start (Not integrated)
# .data
# exp_table:  .word 498, 1353, 3679, 10000  # exp(-3), exp(-2), exp(-1), exp(0) scaled


# .text
# .globl main


# main:
#     la t1, input_values
#     li t2, 10          # number of inputs
#     li t3, 0           # loop counter


#     # Since max input is 0, already known
#     li s1, 0           # max_input = 0


#     # Allocate stack for 10 words
#     addi sp, sp, -40   # 10 * 4 bytes


#     # Load constants for comparison
#     li s2, -3
#     li s3, -2
#     li s4, -1
#     li s5, 0


# compute_exp:
#     beq t3, t2, sum_exp


#     lw t0, 0(t1)       # load input
#     sub t0, t0, s1     # x' = x - max (max=0)


#     # Load exp(x') from table
#     li a0, 0
#     beq t0, s2, load_e3
#     beq t0, s3, load_e2
#     beq t0, s4, load_e1
#     beq t0, s5, load_e0
#     j invalid_input


# load_e3: la a0, exp_table
#          lw a0, 0(a0)
#          j store_exp
# load_e2: la a0, exp_table
#          lw a0, 4(a0)
#          j store_exp
# load_e1: la a0, exp_table
#          lw a0, 8(a0)
#          j store_exp
# load_e0: la a0, exp_table
#          lw a0, 12(a0)
#          j store_exp


# invalid_input:
#     li a0, 0           # fallback for invalid input


# store_exp:
#     sw a0, 0(sp)       # push exp(x') onto stack
#     addi sp, sp, 4
#     addi t1, t1, 4
#     addi t3, t3, 1
#     j compute_exp


# sum_exp:
#     # Reset sp back to base
#     addi sp, sp, -40


#     li t3, 0
#     li s0, 0           # sum_exp = 0
#     addi a1, sp, 0     # a1 = sp (fixed!)


# sum_loop:
#     beq t3, t2, normalize


#     lw a2, 0(a1)
#     add s0, s0, a2
#     addi a1, a1, 4
#     addi t3, t3, 1
#     j sum_loop


# normalize:
#     li t3, 0
#     addi a1, sp, 0     # a1 = sp (fixed!)
#     li s6, 10000       # scale factor


# norm_loop:
#     beq t3, t2, done_norm


#     lw a2, 0(a1)
#     mul a2, a2, s6     # exp(x') * scale
#     div a2, a2, s0     # normalized = exp(x') * scale / sum_exp
#     sw a2, 0(a1)


#     addi a1, a1, 4
#     addi t3, t3, 1
#     j norm_loop


# done_norm:
#     nop
# halt: j halt


# input_values: .word -3, -2, -1, 0, -3, -2, -1, 0, -3, -2


# Softmax end
