/*
 * collect_audio.h
 *
 *  Created on: Nov 4, 2022
 *      Author: mahathir
 */

#ifndef AUDIO_COLLECT_AUDIO_H_
#define AUDIO_COLLECT_AUDIO_H_

#define MIC_POWER_PORT_DIR  P4DIR
#define MIC_POWER_PORT_OUT  P4OUT
#define MIC_POWER_PIN       BIT1

#define AUDIO_PORT_SEL0     P3SEL0
#define AUDIO_PORT_SEL1     P3SEL1
#define MIC_INPUT_PIN       BIT3

#define MIC_INPUT_CHAN      ADC12INCH_15

/*----------------------------------------------------------------------------
 * Defines and Typedefs
 * -------------------------------------------------------------------------*/
typedef struct Audio_Struct *Audio_Handle;

/* The Audio object structure, containing the Audio instance information */
typedef struct Audio_configParams
{
    /* Ping-pong audio buffers
     * To disable ping-pong bufferts, define audioBuffer2 = 0
     */
    int16_t *audioBuffer1;
    int16_t *audioBuffer2;

    /* Size of both audio buffers */
    uint16_t bufferSize;

    /* audio sample rate in Hz */
    uint16_t sampleRate;

    /* Flag indicating DMA is filling audioBuffer1 if true */
    bool fillingBuffer1;

    /* Flag indicating DMA has a filled buffer, and is filling the other */
    bool bufferActive;

    /* Flag indicating that the channel has not processed data in real time */
    bool overflow;
} Audio_configParams;

/*----------------------------------------------------------------------------
* Functions
* --------------------------------------------------------------------------*/
/* Set up the device to collect audio samples in ping-pong buffers */
void Audio_setupCollect(Audio_configParams * audioConfig);

/* Start collecting audio samples in ping-pong buffers */
void Audio_startCollect(Audio_configParams * audioConfig);

// Switch the ping-pong buffer configuration
void Audio_switchBuffer(Audio_configParams * audioConfig);

// Get pointer to the active buffer with valid data ready for processing
int16_t * Audio_getActiveBuffer(Audio_configParams * audioConfig);

// Get pointer to the current buffer where data is currently written
int16_t * Audio_getCurrentBuffer(Audio_configParams * audioConfig);

/* Check if a frame of data is ready for processing */
bool Audio_getActive(Audio_configParams *audioConfig);

/* Indicate done with processing active buffer holding valid data */
void Audio_resetActive(Audio_configParams * audioConfig);

/* Get overflow error status */
bool Audio_getOverflow(Audio_configParams * audioConfig);

/* Reset overflow error status */
void Audio_resetOverflow(Audio_configParams * audioConfig);

/* Stop collecting audio samples in buffers */
void Audio_stopCollect(Audio_configParams * audioConfig);

/* Shut down the audio collection peripherals */
void Audio_shutdownCollect(Audio_configParams * audioConfig);

#endif /* AUDIO_COLLECT_AUDIO_H_ */
