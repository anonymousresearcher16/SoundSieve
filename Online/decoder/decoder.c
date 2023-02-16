/*
 * decoder.c
 * This file decodes the model array encoded by encoder.py and translate the data into the configuration of layers
 */

#include "decoder.h"

/* used to save the results of each layer and prevent data overwrite */
#pragma LOCATION(MODEL_ARRAY_OUTPUT, 0x10000)
#pragma PERSISTENT(MODEL_ARRAY_OUTPUT)
static int16_t MODEL_ARRAY_OUTPUT[MODEL_ARRAY_OUTPUT_LENGTH] = {0};

#pragma PERSISTENT(MODEL_ARRAY_TEMP)
static int16_t MODEL_ARRAY_TEMP[MODEL_ARRAY_TEMP_LENGTH] = {0};

#pragma LOCATION(MODEL_ARRAY_TEMP2, 0x22000)
#pragma PERSISTENT(MODEL_ARRAY_TEMP2)
static int16_t MODEL_ARRAY_TEMP2[MODEL_ARRAY_TEMP_LENGTH] = {0};



matrix *apply_model(matrix *output, matrix *input){

    unsigned long int addr = 0x2b000;
    unsigned long int row_addr = 0x2b002;
    unsigned long int col_addr = 0x2b004;
    unsigned long int conv_addr = 0x2b006;
    unsigned long int half_addr = 0x2b008;

    int16_t *array = MODEL_ARRAY;
    int16_t *bias_array;
    uint16_t cnt = 0;
    uint16_t conv_id = 0;
    uint16_t half_id = 0;
    uint16_t stop_flag = 0;
    uint16_t curr_layer = 0;


    uint16_t layer_class, activation, numChannels, filter_numRows, filter_numCols, stride_numRows, stride_numCols, filters_length, padding;
    uint16_t numFilters;
    output->data = MODEL_ARRAY_OUTPUT;

    int16_t INDEX = __data20_read_short(addr);
    if (INDEX == -1)
        INDEX = 0;

    array += INDEX;
    cnt += INDEX;

    int16_t CONV_INDEX = __data20_read_short(conv_addr);
    if (CONV_INDEX == -1)
        CONV_INDEX = 0;
    conv_id += CONV_INDEX;

    int16_t HALF_INDEX = __data20_read_short(half_addr);
    if (HALF_INDEX == -1)
        HALF_INDEX = 0;
    half_id += HALF_INDEX;

    // Sequential model
    if (INDEX == 0){  // 1st element of the array tells the model type
        array ++;
        cnt ++;
    }
    else{
        input->data = MODEL_ARRAY_TEMP;
        input->numRows = __data20_read_short(row_addr);
        input->numCols = __data20_read_short(col_addr);
        if ((conv_id == 1) && (half_id == 1)){
            input->data = input_buffer;
        }
    }
    uint16_t dense_cnt = 0;

    while (array < MODEL_ARRAY_END){
        // next element of the array tells the layer class

        /* layer class 0 - DENSE */
        if (*array == DENSE_LAYER){
            dense_cnt ++;
            curr_layer = 1;
            numFilters = 1;

            // extract and prepare layer parameters
            layer_class = *array;
            activation = *(array + 1);
            uint16_t kernel_numRows = *(array + 2);
            uint16_t kernel_numCols = *(array + 3);
            uint16_t bias_numRows = *(array + 4);
            uint16_t bias_numCols = *(array + 5);
            array += 6;
            cnt += 6;
            uint16_t kernel_length = kernel_numRows * kernel_numCols;
            uint16_t bias_length = bias_numRows * bias_numCols;

            // extract layer weights
            int16_t *kernel_array = array;
            array += kernel_length;
            cnt += kernel_length;

            bias_array = array;
            array += bias_length;
            cnt += bias_length;

            // prepare output
            uint16_t output_numRows = kernel_numRows;
            uint16_t output_numCols = input->numCols;
            output->numRows = output_numRows;
            output->numCols = output_numCols;

            // initialize weight matrix
            matrix kernel = {kernel_array, kernel_numRows, kernel_numCols};
            matrix bias = {bias_array, bias_numRows, bias_numCols};

            // execute dense layer
            if (activation == RELU_ACTIVATION){
                dense(output, input, &kernel, &bias, &fp_relu, FIXED_POINT_PRECISION);
            }
            else if (activation == SIGMOID_ACTIVATION){
                dense(output, input, &kernel, &bias, &fp_sigmoid, FIXED_POINT_PRECISION);
            }
            else{
                dense(output, input, &kernel, &bias, &fp_linear, FIXED_POINT_PRECISION);
            }
            stop_flag = 1;
        }

        /* layer class 1 - LeakyReLU */
        else if (*array == LEAKY_RELU_LAYER){
            curr_layer = 2;
            output->numRows = input->numRows;
            output->numCols = input->numCols;
            apply_leakyrelu(output, input, FIXED_POINT_PRECISION);
            array ++;
            cnt ++;
        }

        /* layer class 2 - Conv2D */
        else if (*array == CONV2D_LAYER){
            curr_layer = 3;
            if (half_id == 2){
                half_id = 0;
                conv_id += 1;
            }
            else if (half_id == 0){
                conv_id += 1;
            }

            // extract and prepare layer parameters
            layer_class = *array;
            activation = *(array + 1);


            if (conv_id == 1){
                numFilters = *(array + 2);
                //numFilters = *(array + 2) / 2;
            }
            else if (conv_id == 2){
                numFilters = *(array + 2);
                //numFilters = *(array + 2) / 2;
            }
            else if (conv_id == 3){
                //numFilters = *(array + 2);
                numFilters = *(array + 2) / 2;
            }



            numChannels = *(array + 3);
            filter_numRows = *(array + 4);
            filter_numCols = *(array + 5);
            stride_numRows = *(array + 6);
            stride_numCols = *(array + 7);

            if (conv_id == 1){
                filters_length = *(array + 8);
                //filters_length = *(array + 8) / 2;
            }
            else if (conv_id == 2){
                filters_length = *(array + 8);
                //filters_length = *(array + 8) / 2;
            }
            else if (conv_id == 3){
                //filters_length = *(array + 8);
                filters_length = *(array + 8) / 2;
            }


            padding = *(array + 9);
            //if (!((conv_id == 1) || (conv_id == 2) || (conv_id == 3))){
            if (!(conv_id == 3)){
                array += 10;
                cnt += 10;
            }


            // prepare output
            if (padding == 1){
                output->numRows = input->numRows / stride_numRows;
                if (input->numRows % stride_numRows > 0){
                    output->numRows ++;
                }
                output->numCols = input->numCols / stride_numCols;
                if (input->numCols % stride_numRows > 0){
                    output->numCols ++;
                }
            }
            else {
                output->numRows = (input->numRows - filter_numRows + 1) / stride_numRows;
                if ((input->numRows - filter_numRows + 1) % stride_numRows > 0){
                    output->numRows ++;
                }
                output->numCols = (input->numCols - filter_numCols + 1) / stride_numCols;
                if ((input->numCols - filter_numCols + 1) % stride_numCols > 0){
                    output->numCols ++;
                }
            }

            // extract and prepare weights
            int16_t *filters_array;

            if (conv_id == 1){
                filters_array = array;
                /*
                if (half_id == 0){
                    filters_array = array + 10;
                }
                else{
                    filters_array = array + filters_length + 10;
                }
                */

            }
            else if (conv_id == 2){
                filters_array = array;
                /*
                if (half_id == 0){
                    filters_array = array + 10;
                }
                else{
                    filters_array = array + filters_length + 10;
                }
                */
            }
            else if (conv_id == 3){
                //filters_array = array;

                if (half_id == 0){
                    filters_array = array + 10;
                }
                else{
                    filters_array = array + filters_length + 10;
                }
            }


            matrix filters = {filters_array, filter_numRows, filter_numCols};
            if (conv_id == 1){
                array += filters_length;
                cnt += filters_length;
                /*
                if (half_id == 1){
                    array += (filters_length * 2 + 10);
                    cnt += (filters_length * 2 + 10);
                }
                */
            }
            else if (conv_id == 2){
                array += filters_length;
                cnt += filters_length;
                /*
                if (half_id == 1){
                    array += (filters_length * 2 + 10);
                    cnt += (filters_length * 2 + 10);
                }
                */
            }
            else if (conv_id == 3){
                //array += filters_length;
                //cnt += filters_length;
                if (half_id == 1){
                    array += (filters_length * 2 + 10);
                    cnt += (filters_length * 2 + 10);
                }
            }


            if (conv_id == 1){
                bias_array = array;
                /*
                if (half_id == 1){
                    bias_array = array + numFilters;
                }
                else{
                    bias_array = array + filters_length * 2 + 10;
                }
                */
            }
            else if (conv_id == 2){
                bias_array = array;
                /*
                if (half_id == 1){
                    bias_array = array + numFilters;
                }
                else{
                    bias_array = array + filters_length * 2 + 10;
                }
                */
            }
            else if (conv_id == 3){
                //bias_array = array;
                if (half_id == 1){
                    bias_array = array + numFilters;
                }
                else{
                    bias_array = array + filters_length * 2 + 10;
                }
            }


            if (conv_id == 1){
                array += numFilters;
                cnt += numFilters;
                /*
                if (half_id == 1){
                    array += numFilters * 2;
                    cnt += numFilters * 2;
                }
                */
            }
            else if (conv_id == 2){
                array += numFilters;
                cnt += numFilters;
                /*
                if (half_id == 1){
                    array += numFilters * 2;
                    cnt += numFilters * 2;
                }
                */
            }
            else if (conv_id == 3){
                //array += numFilters;
                //cnt += numFilters;

                if (half_id == 1){
                    array += numFilters * 2;
                    cnt += numFilters * 2;
                }
            }
            //array += numFilters;
            //cnt += numFilters;


            // execute conv2d layer
            if (activation == RELU_ACTIVATION){
                conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_relu, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
            }
            else if (activation == SIGMOID_ACTIVATION){
                conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_sigmoid, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
            }
            else{
                conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_linear, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
            }
            //if ((conv_id == 1) || (conv_id == 2) || (conv_id == 3))
            if (conv_id == 3)
                half_id += 1;

            if (half_id == 1)
                stop_flag = 1;


        }

        /* layer class 3 - MaxPooling2D */
        else if (*array == MAXPOOLING2D_LAYER){
            curr_layer = 4;
            uint16_t pool_numRows = *(array + 1);
            uint16_t pool_numCols = *(array + 2);
            stride_numRows = *(array + 3);
            stride_numCols = *(array + 4);
            padding = *(array + 5);
            array += 6;
            cnt += 6;

            output->numRows = input->numRows / pool_numRows;
            output->numCols = input->numCols / pool_numCols;

            maxpooling_filters(output, input, numFilters, pool_numRows, pool_numCols);
            if (conv_id != 3)
                stop_flag = 1;
        }

        /* layer class 4 - Conv2D Flatten */
        else if (*array == FLATTEN_LAYER){
            curr_layer = 5;
            array += 1;
            cnt += 1;
            output->numRows = input->numRows * input->numCols * numFilters;
            output->numCols = LEA_RESERVED;
            flatten(output, input, numFilters);
            numFilters = 1;
            stop_flag = 1;
        }
        /* SKIP FOR INFERENCE TIME IMPLEMENTATION - layer class 5 - Dropout Layer */
        else if (*array == DROPOUT_LAYER){
            curr_layer = 6;
            array += 1;
            cnt += 1;
            numFilters = 1;
        }

        //INDEX = cnt;

        __data20_write_short(addr, cnt);

        /* copy output matrix and reference input to copied output */
        if (curr_layer == 3){
            if (conv_id == 1){
                /*
                if (half_id == 2){
                    dma_load(MODEL_ARRAY_TEMP2 + output->numRows * output->numCols * numFilters, output->data, output->numRows * output->numCols * numFilters);
                    numFilters = numFilters * 2;
                    filters_length = filters_length * 2;
                }
                else{
                    dma_load(MODEL_ARRAY_TEMP2, output->data, output->numRows * output->numCols * numFilters);
                }
                */
                dma_load(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters);
            }
            else if (conv_id == 2){
                /*
                if (half_id == 2){
                    dma_load(MODEL_ARRAY_TEMP2 + output->numRows * output->numCols * numFilters, output->data, output->numRows * output->numCols * numFilters);
                    numFilters = numFilters * 2;
                    filters_length = filters_length * 2;
                }
                else{
                    dma_load(MODEL_ARRAY_TEMP2, output->data, output->numRows * output->numCols * numFilters);
                }
                */
                dma_load(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters);
            }
            else if (conv_id == 3){
                if (half_id == 2){
                    dma_load(MODEL_ARRAY_TEMP2 + output->numRows * output->numCols * numFilters, output->data, output->numRows * output->numCols * numFilters);
                    numFilters = numFilters * 2;
                    filters_length = filters_length * 2;
                }
                else{
                    dma_load(MODEL_ARRAY_TEMP2, output->data, output->numRows * output->numCols * numFilters);
                }
                //dma_load(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters);
            }
        }
        else
            dma_load(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters);
        //dma_load(MODEL_ARRAY_TEMP2, output->data, output->numRows * output->numCols * numFilters);
        //if (((conv_id == 1) || (conv_id == 2) || (conv_id == 3)) && (half_id == 2) && (curr_layer == 3)){
        if ((conv_id == 3) && (half_id == 2) && (curr_layer == 3)){
            input->data = MODEL_ARRAY_TEMP2;
            input->numRows = output->numRows;
            input->numCols = output->numCols;
            output->data = MODEL_ARRAY_TEMP2;
        }
        //else if (!( ((conv_id == 1) && (half_id == 1)) || ((conv_id == 2) && (half_id == 1)) || ((conv_id == 3) && (half_id == 1)) )){
        else if (!((conv_id == 3) && (half_id == 1))){
            input->data = MODEL_ARRAY_TEMP;
            input->numRows = output->numRows;
            input->numCols = output->numCols;
            //output->data = MODEL_ARRAY_TEMP;
            output->data = MODEL_ARRAY_OUTPUT;
        }




        __data20_write_short(row_addr, input->numRows);
        __data20_write_short(col_addr, input->numCols);
        __data20_write_short(conv_addr, conv_id);
        __data20_write_short(half_addr, half_id);

        if (stop_flag == 1)
            return output;
            //stop_flag = 0;
    }


    __data20_write_short(addr, -1);
    __data20_write_short(row_addr, -1);
    __data20_write_short(col_addr, -1);
    __data20_write_short(conv_addr, -1);
    __data20_write_short(half_addr, -1);

    label = argmax(output);

    unsigned long int label_addr = 0x2b010;
    __data20_write_short(label_addr, label);

    return output;
}
