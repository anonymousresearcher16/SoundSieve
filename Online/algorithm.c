/*
 * algorithm.c
 *
 *  Created on: Nov 4, 2022
 *      Author: mahathir
 */

#include "global.h"

#ifndef NEURAL_NETWORK_PARAMS_GUARD
#include "neural_network_parameters.h"
#endif

#ifndef PREDICTOR_NEURAL_NETWORK_PARAMS_GUARD
#include "predictor_neural_network_parameters.h"
#endif

#ifndef DECODER_GUARD
#include "decoder/decoder.h"
#endif

#define AUDIO_LENGTH            1 // 1 second
#define SAMPLE_RATE             2560 // Hz
#define SIGNAL_LENGTH           AUDIO_LENGTH*SAMPLE_RATE
#define FRAME_SIZE              0.05 // 50 ms
#define FRAME_STRIDE            0.025 // 25 ms
#define FRAME_LENGTH            SAMPLE_RATE*FRAME_SIZE // 128
#define FRAME_STEP              SAMPLE_RATE*FRAME_STRIDE // 64
#define NUM_FRAMES              40 // (SIGNAL_LENGTH - FRAME_LENGTH) / FRAME_STEP = 60.5 -> round up to 60
#define NBINS                   64
#define NUM_FRAMES_PER_SEGMENT 3
#define NBIMS_CONV 32
#define FEAT_SIZE NUM_FRAMES_PER_SEGMENT * NBIMS_CONV * 2

#define VECTOR_SIZE             128
#define TWO_VECTOR_SIZE         256
#define HALF_VECTOR_SIZE        64

#define FFT_SAMPLING_FREQUENCY  2560


typedef union
{
    _iq31 raw[946];
    struct FftData_s
    {
        _q15 input[TWO_VECTOR_SIZE];
        _q15 samples[TWO_VECTOR_SIZE];
        _q15 sourceA[VECTOR_SIZE];
        _q15 sourceB[VECTOR_SIZE];
        _q15 dest[VECTOR_SIZE];
        _iq31 result;
    } fftDataParam;
} LeaMem;

#pragma DATA_SECTION(leaMem, ".leaRAM")
LeaMem leaMem;


#pragma PERSISTENT(spectrum)
_iq24 spectrum[NUM_FRAMES][NBINS] = { 0, };

#pragma PERSISTENT(segment_spectrum)
_iq24 segment_spectrum[NUM_FRAMES_PER_SEGMENT][NBIMS_CONV] = { 0, };


// Global audio config parameter
Audio_configParams gAudioConfig;

// Allocate DSPLib structures
static msp_status status;
static msp_mpy_q15_params mpyParams;
static msp_cmplx_fft_q15_params fftParams;
static msp_cmplx_fill_q15_params fillParams;
static msp_interleave_q15_params interleaveParams;

const _q15 hammingWindow[VECTOR_SIZE] = {
    _Q15(0.0800), _Q15(0.0801), _Q15(0.0806), _Q15(0.0813),
    _Q15(0.0822), _Q15(0.0835), _Q15(0.0850), _Q15(0.0868),
    _Q15(0.0889), _Q15(0.0913), _Q15(0.0939), _Q15(0.0968),
    _Q15(0.1000), _Q15(0.1034), _Q15(0.1071), _Q15(0.1111),
    _Q15(0.1153), _Q15(0.1198), _Q15(0.1245), _Q15(0.1295),
    _Q15(0.1347), _Q15(0.1402), _Q15(0.1459), _Q15(0.1519),
    _Q15(0.1581), _Q15(0.1645), _Q15(0.1712), _Q15(0.1781),
    _Q15(0.1852), _Q15(0.1925), _Q15(0.2001), _Q15(0.2078),
    _Q15(0.2157), _Q15(0.2239), _Q15(0.2322), _Q15(0.2407),
    _Q15(0.2494), _Q15(0.2583), _Q15(0.2673), _Q15(0.2765),
    _Q15(0.2859), _Q15(0.2954), _Q15(0.3051), _Q15(0.3149),
    _Q15(0.3249), _Q15(0.3350), _Q15(0.3452), _Q15(0.3555),
    _Q15(0.3659), _Q15(0.3765), _Q15(0.3871), _Q15(0.3979),
    _Q15(0.4087), _Q15(0.4196), _Q15(0.4305), _Q15(0.4416),
    _Q15(0.4527), _Q15(0.4638), _Q15(0.4750), _Q15(0.4863),
    _Q15(0.4976), _Q15(0.5089), _Q15(0.5202), _Q15(0.5315),
    _Q15(0.5428), _Q15(0.5542), _Q15(0.5655), _Q15(0.5768),
    _Q15(0.5881), _Q15(0.5993), _Q15(0.6106), _Q15(0.6217),
    _Q15(0.6329), _Q15(0.6439), _Q15(0.6549), _Q15(0.6659),
    _Q15(0.6767), _Q15(0.6875), _Q15(0.6982), _Q15(0.7088),
    _Q15(0.7193), _Q15(0.7297), _Q15(0.7400), _Q15(0.7501),
    _Q15(0.7601), _Q15(0.7700), _Q15(0.7797), _Q15(0.7893),
    _Q15(0.7988), _Q15(0.8081), _Q15(0.8172), _Q15(0.8262),
    _Q15(0.8350), _Q15(0.8436), _Q15(0.8520), _Q15(0.8602),
    _Q15(0.8683), _Q15(0.8761), _Q15(0.8837), _Q15(0.8912),
    _Q15(0.8984), _Q15(0.9054), _Q15(0.9121), _Q15(0.9187),
    _Q15(0.9250), _Q15(0.9311), _Q15(0.9369), _Q15(0.9426),
    _Q15(0.9479), _Q15(0.9530), _Q15(0.9579), _Q15(0.9625),
    _Q15(0.9669), _Q15(0.9710), _Q15(0.9748), _Q15(0.9784),
    _Q15(0.9817), _Q15(0.9847), _Q15(0.9875), _Q15(0.9899),
    _Q15(0.9922), _Q15(0.9941), _Q15(0.9958), _Q15(0.9972),
    _Q15(0.9983), _Q15(0.9991), _Q15(0.9997), _Q15(0.9999),
    _Q15(0.9999), _Q15(0.9997), _Q15(0.9991), _Q15(0.9983),
    _Q15(0.9972), _Q15(0.9958), _Q15(0.9941), _Q15(0.9922),
    _Q15(0.9899), _Q15(0.9875), _Q15(0.9847), _Q15(0.9817),
    _Q15(0.9784), _Q15(0.9748), _Q15(0.9710), _Q15(0.9669),
    _Q15(0.9625), _Q15(0.9579), _Q15(0.9530), _Q15(0.9479),
    _Q15(0.9426), _Q15(0.9369), _Q15(0.9311), _Q15(0.9250),
    _Q15(0.9187), _Q15(0.9121), _Q15(0.9054), _Q15(0.8984),
    _Q15(0.8912), _Q15(0.8837), _Q15(0.8761), _Q15(0.8683),
    _Q15(0.8602), _Q15(0.8520), _Q15(0.8436), _Q15(0.8350),
    _Q15(0.8262), _Q15(0.8172), _Q15(0.8081), _Q15(0.7988),
    _Q15(0.7893), _Q15(0.7797), _Q15(0.7700), _Q15(0.7601),
    _Q15(0.7501), _Q15(0.7400), _Q15(0.7297), _Q15(0.7193),
    _Q15(0.7088), _Q15(0.6982), _Q15(0.6875), _Q15(0.6767),
    _Q15(0.6659), _Q15(0.6549), _Q15(0.6439), _Q15(0.6329),
    _Q15(0.6217), _Q15(0.6106), _Q15(0.5993), _Q15(0.5881),
    _Q15(0.5768), _Q15(0.5655), _Q15(0.5542), _Q15(0.5428),
    _Q15(0.5315), _Q15(0.5202), _Q15(0.5089), _Q15(0.4976),
    _Q15(0.4863), _Q15(0.4750), _Q15(0.4638), _Q15(0.4527),
    _Q15(0.4416), _Q15(0.4305), _Q15(0.4196), _Q15(0.4087),
    _Q15(0.3979), _Q15(0.3871), _Q15(0.3765), _Q15(0.3659),
    _Q15(0.3555), _Q15(0.3452), _Q15(0.3350), _Q15(0.3249),
    _Q15(0.3149), _Q15(0.3051), _Q15(0.2954), _Q15(0.2859),
    _Q15(0.2765), _Q15(0.2673), _Q15(0.2583), _Q15(0.2494),
    _Q15(0.2407), _Q15(0.2322), _Q15(0.2239), _Q15(0.2157),
    _Q15(0.2078), _Q15(0.2001), _Q15(0.1925), _Q15(0.1852),
    _Q15(0.1781), _Q15(0.1712), _Q15(0.1645), _Q15(0.1581),
    _Q15(0.1519), _Q15(0.1459), _Q15(0.1402), _Q15(0.1347),
    _Q15(0.1295), _Q15(0.1245), _Q15(0.1198), _Q15(0.1153),
    _Q15(0.1111), _Q15(0.1071), _Q15(0.1034), _Q15(0.1000),
    _Q15(0.0968), _Q15(0.0939), _Q15(0.0913), _Q15(0.0889),
    _Q15(0.0868), _Q15(0.0850), _Q15(0.0835), _Q15(0.0822),
    _Q15(0.0813), _Q15(0.0806), _Q15(0.0801), _Q15(0.0800)
};

static void copyData(const void *src, void *dst, uint16_t length)
{
    uint16_t i;
    uint16_t *srcPtr;
    uint16_t *dstPtr;

    // Set src and dst pointers
    srcPtr = (uint16_t *) src;
    dstPtr = (uint16_t *) dst;

    for (i = 0; i < length / 2; i++) *dstPtr++ = *srcPtr++;
}

static void initAudio(void)
{
    // Initialize the microphone for recording
    gAudioConfig.audioBuffer1 = (int16_t *) &leaMem.fftDataParam.samples[0];
    gAudioConfig.audioBuffer2 = (int16_t *) &leaMem.fftDataParam.samples[VECTOR_SIZE];
    gAudioConfig.bufferSize = VECTOR_SIZE;
    gAudioConfig.sampleRate = FFT_SAMPLING_FREQUENCY;
    Audio_setupCollect(&gAudioConfig);

    // Start the recording by enabling the timer
    Audio_startCollect(&gAudioConfig);
}

static void deinitAudio(void)
{
    // Stop the audio collection
    Audio_stopCollect(&gAudioConfig);
    Audio_shutdownCollect(&gAudioConfig);
}

static void get_fft(int16_t *inputPtr, int16_t *bufferPtr)
{
    // Set the input and buffer pointers
    inputPtr = leaMem.fftDataParam.input;
    bufferPtr = Audio_getActiveBuffer(&gAudioConfig);

    // Apply Hamming window to buffer samples
    status = msp_mpy_q15(&mpyParams, bufferPtr, leaMem.fftDataParam.sourceB, bufferPtr);
    if (status != MSP_SUCCESS) P1OUT |= BIT0;

    // Zero fill FFT input buffer
    status = msp_cmplx_fill_q15(&fillParams, inputPtr);
    if (status != MSP_SUCCESS)P1OUT |= BIT0;

    // Interleave input samples to real component of FFT input
    status = msp_interleave_q15(&interleaveParams, bufferPtr, inputPtr);
    if (status != MSP_SUCCESS)P1OUT |= BIT0;

    // Run FFT on complex input
    status = msp_cmplx_fft_fixed_q15(&fftParams, inputPtr);
    if (status != MSP_SUCCESS)P1OUT |= BIT0;
}

static void get_powerspectrum(int16_t *inputPtr, uint16_t *powerspectrum)
{
    uint16_t i;
    uint16_t real_abs;
    uint16_t imag_abs;

    for (i = 0; i < HALF_VECTOR_SIZE; i++) //Should it be VECTOR_SIZE or HALF_VECTOR_SIZE
    {
        // Approximate magnitude for the positive frequency domain
        real_abs = abs(inputPtr[2 * i + 0]);
        imag_abs = abs(inputPtr[2 * i + 1]);

        //power[i] = (real_abs*real_abs + imag_abs*imag_abs) / VECTOR_SIZE;
        powerspectrum[i] = real_abs * real_abs + imag_abs * imag_abs;
    }
}

static uint16_t predict(_iq24 *feature)
{
    copyData(feature, predictor_input_buffer, FEAT_SIZE);

    predictor_inputFeatures.numRows = PREDICTOR_INPUT_NUM_ROWS;
    predictor_inputFeatures.numCols = PREDICTOR_INPUT_NUM_COLS;
    predictor_inputFeatures.data = predictor_input_buffer;


    predictor_outputLabels.numRows = PREDICTOR_OUTPUT_NUM_LABELS;
    predictor_outputLabels.numCols = LEA_RESERVED;   // one more column is reserved for LEA
    predictor_outputLabels.data = predictor_output_buffer;

   apply_model(&predictor_outputLabels, &predictor_inputFeatures);
}

static void get_feature(int16_t *data, _iq24 *feature, _iq24 *segment_feature)
{
    int16_t *inputPtr = leaMem.fftDataParam.input;

    get_fft(inputPtr, data);
    get_powerspectrum(inputPtr, feature);
    copyData(feature, segment_feature, VECTOR_SIZE);
}

void init_infer(void)
{
    // Initialize multiply parameters and copy Hamming window
    mpyParams.length = VECTOR_SIZE;
    copyData(hammingWindow, leaMem.fftDataParam.sourceB, sizeof(hammingWindow));

    // Initialize complex fill parameters
    fillParams.length = VECTOR_SIZE;
    fillParams.realValue = 0;
    fillParams.imagValue = 0;

    // Initialize interleave parameters
    interleaveParams.length = VECTOR_SIZE;
    interleaveParams.channel = 0;
    interleaveParams.numChannels = 2;

    // Initialize FFT parameters
    fftParams.length = VECTOR_SIZE;
    fftParams.bitReverse = 1;
    fftParams.twiddleTable = msp_cmplx_twiddle_table_256_q15;
}

uint16_t infer(void)
{
    int16_t *audio_data1, *audio_data2;
    int16_t *data;

    uint16_t frame_num = 0, segment_frame_num=0;

    initAudio();

    while (1)
    {
        //__bis_SR_register(LPM3_bits + GIE);

        // wake up and run when audio data is ready!
        if (frame_num > 0) {

            if (gAudioConfig.fillingBuffer1)
            {
                audio_data1 = gAudioConfig.audioBuffer1;
                audio_data2 = gAudioConfig.audioBuffer2;
            }
            else
            {
                audio_data1 = gAudioConfig.audioBuffer2;
                audio_data2 = gAudioConfig.audioBuffer1;
            }

            // overlapped
            copyData(&audio_data1[HALF_VECTOR_SIZE], leaMem.fftDataParam.dest, TWO_VECTOR_SIZE);
            copyData(&audio_data2[0],&leaMem.fftDataParam.dest[HALF_VECTOR_SIZE], TWO_VECTOR_SIZE);
            data = leaMem.fftDataParam.dest;

            get_feature(data, spectrum[frame_num++], segment_spectrum[segment_frame_num++]);


            if (frame_num >= NUM_FRAMES)
            {
                frame_num = 0;
                break;
            }

            if (segment_frame_num >= NUM_FRAMES)
            {
                segment_frame_num = 0;
                predict(segment_spectrum);
                check_voltage();
                wait_in_lpm();
            }


        }

        data = Audio_getActiveBuffer(&gAudioConfig);
        get_feature(data, spectrum[frame_num++], segment_spectrum[segment_frame_num++]);

        if (frame_num >= NUM_FRAMES)
        {
            frame_num = 0;
            break;
        }

        if (segment_frame_num >= NUM_FRAMES)
        {
            segment_frame_num = 0;
            predict(segment_spectrum);
            check_voltage();
            wait_in_lpm();
        }
    }

    deinitAudio();

    return 1;
}
