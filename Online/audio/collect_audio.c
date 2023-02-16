/*
 * collect_audio.c
 *
 *  Created on: Nov 4, 2022
 *      Author: mahathir
 */

#include <global.h>

/* Function that powers up the external microphone and starts sampling
 * the microphone output.
 * The ADC is triggered to sample using the Timer module
 * Then the data is moved via DMA. The device would only wake-up once
 * the DMA is done. */

void Audio_setupCollect(Audio_configParams * audioConfig)
{
    DMA_initParam dmaConfig;
    ADC12_B_initParam adcConfig;
    ADC12_B_configureMemoryParam adcMemConfig;
    Timer_A_initUpModeParam timerUpConfig;
    Timer_A_initCompareModeParam timerCompareConfig;

    // Configure TA1.1 for ADC sample interval
    timerUpConfig.clockSource = TIMER_A_CLOCKSOURCE_SMCLK;
    timerUpConfig.clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_1;
    timerUpConfig.timerPeriod = (__SYSTEM_FREQUENCY_MHZ__ / audioConfig->sampleRate) - 1;
    timerUpConfig.timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_DISABLE;
    timerUpConfig.captureCompareInterruptEnable_CCR0_CCIE = TIMER_A_CCIE_CCR0_INTERRUPT_DISABLE;
    timerUpConfig.timerClear = TIMER_A_DO_CLEAR;
    timerUpConfig.startTimer = false;
    Timer_A_initUpMode(TIMER_A0_BASE, &timerUpConfig);

    // Initialize TA0CCR1 to generate trigger clock output, reset/set mode
    timerCompareConfig.compareRegister = TIMER_A_CAPTURECOMPARE_REGISTER_1;
    timerCompareConfig.compareInterruptEnable = TIMER_A_CAPTURECOMPARE_INTERRUPT_DISABLE;
    timerCompareConfig.compareOutputMode = TIMER_A_OUTPUTMODE_SET_RESET;
    timerCompareConfig.compareValue = ((__SYSTEM_FREQUENCY_MHZ__ / audioConfig->sampleRate) / 2) - 1;
    Timer_A_initCompareMode(TIMER_A0_BASE, &timerCompareConfig);

    // Turn on microphone preamp
    GPIO_setAsOutputPin(GPIO_PORT_P4, GPIO_PIN1);
    GPIO_setOutputHighOnPin(GPIO_PORT_P4, GPIO_PIN1);

    // Configure ADC triggered by TA1.1
    GPIO_setAsPeripheralModuleFunctionOutputPin(GPIO_PORT_P3, GPIO_PIN0, GPIO_TERNARY_MODULE_FUNCTION);
    adcConfig.sampleHoldSignalSourceSelect = ADC12_B_SAMPLEHOLDSOURCE_1;
    adcConfig.clockSourceSelect = ADC12_B_CLOCKSOURCE_ADC12OSC;
    adcConfig.clockSourceDivider = ADC12_B_CLOCKDIVIDER_1;
    adcConfig.clockSourcePredivider = ADC12_B_CLOCKPREDIVIDER__1;
    adcConfig.internalChannelMap = ADC12_B_NOINTCH;
    ADC12_B_init(ADC12_B_BASE, &adcConfig);
    ADC12_B_enable(ADC12_B_BASE);
    ADC12_B_setupSamplingTimer(ADC12_B_BASE, ADC12_B_CYCLEHOLD_16_CYCLES, ADC12_B_CYCLEHOLD_16_CYCLES, ADC12_B_MULTIPLESAMPLESDISABLE);
    ADC12_B_setResolution(ADC12_B_BASE, ADC12_B_RESOLUTION_12BIT);
    ADC12_B_setDataReadBackFormat(ADC12_B_BASE, ADC12_B_SIGNED_2SCOMPLEMENT);

    adcMemConfig.differentialModeSelect = ADC12_B_DIFFERENTIAL_MODE_DISABLE;
    adcMemConfig.endOfSequence = ADC12_B_NOTENDOFSEQUENCE;
    adcMemConfig.inputSourceSelect = ADC12_B_INPUT_A15;
    adcMemConfig.memoryBufferControlIndex = ADC12_B_MEMORY_0;
    adcMemConfig.refVoltageSourceSelect = ADC12_B_VREFPOS_AVCC_VREFNEG_VSS;
    adcMemConfig.windowComparatorSelect = ADC12_B_WINDOW_COMPARATOR_DISABLE;
    ADC12_B_configureMemory(ADC12_B_BASE, &adcMemConfig);
    ADC12_B_startConversion(ADC12_B_BASE, ADC12_B_MEMORY_0, ADC12_B_REPEATED_SINGLECHANNEL);

    // Start with primary (first) audio buffer transfer
    audioConfig->fillingBuffer1 = true;
    audioConfig->overflow = false;
    audioConfig->bufferActive = false;

    // Setup DMA0 transfer to memory
    dmaConfig.channelSelect = DMA_CHANNEL_0;
    dmaConfig.transferModeSelect = DMA_TRANSFER_SINGLE;
    dmaConfig.transferSize = audioConfig->bufferSize;
    dmaConfig.triggerSourceSelect = DMA0TSEL__ADC12IFG;
    dmaConfig.transferUnitSelect = DMA_SIZE_SRCWORD_DSTWORD;
    dmaConfig.triggerTypeSelect = DMA_TRIGGER_RISINGEDGE;
    DMA_init(&dmaConfig);
    DMA_setSrcAddress(DMA_CHANNEL_0, (uint32_t) &ADC12MEM0, DMA_DIRECTION_UNCHANGED);
    DMA_setDstAddress(DMA_CHANNEL_0,
                      (uint32_t) audioConfig->audioBuffer1,
                      DMA_DIRECTION_INCREMENT);

    DMA_disableTransferDuringReadModifyWrite();
    DMA_clearInterrupt(DMA_CHANNEL_0);
    DMA_enableInterrupt(DMA_CHANNEL_0);
    DMA_enableTransfers(DMA_CHANNEL_0);

    // Turn on mic power full drive strength and enable mic input pin to ADC
    MIC_POWER_PORT_OUT |= MIC_POWER_PIN;
    MIC_POWER_PORT_DIR |= MIC_POWER_PIN;
    AUDIO_PORT_SEL0 |= MIC_INPUT_PIN;
    AUDIO_PORT_SEL1 |= MIC_INPUT_PIN;

    // Delay for 1ms to settle microphone
    __delay_cycles(__SYSTEM_FREQUENCY_MHZ__/1000);

}


/*--------------------------------------------------------------------------*/
/* Start collecting audio samples in ping-pong buffers */
void Audio_startCollect(Audio_configParams * audioConfig)
{
    // Start TA0 timer to begin audio data collection
    Timer_A_clear(TIMER_A0_BASE);
    Timer_A_startCounter(TIMER_A0_BASE, TIMER_A_UP_MODE);
}

/*--------------------------------------------------------------------------*/
/* Switch buffers collecting audio samples in ping-pong buffers */
void Audio_switchBuffer(Audio_configParams * audioConfig)
{
    // Check if it is using only single or dual buffer
    if(audioConfig->audioBuffer2 != 0)
    {
        if(audioConfig->fillingBuffer1)
        {
            DMA_setDstAddress(DMA_CHANNEL_0,
                              (uint32_t) audioConfig->audioBuffer2,
                              DMA_DIRECTION_INCREMENT);
            audioConfig->fillingBuffer1 = false;
        }
        else
        {
            DMA_setDstAddress(DMA_CHANNEL_0,
                              (uint32_t) audioConfig->audioBuffer1,
                              DMA_DIRECTION_INCREMENT);
            audioConfig->fillingBuffer1 = true;
        }
    }

    // Most likely an overflow condition has occurred
    if(audioConfig->bufferActive)
    {
        audioConfig->overflow = true;
    }

    // Enable the DMA0 to start receiving triggers when ADC sample available
    DMA_enableTransfers(DMA_CHANNEL_0);
}

/*--------------------------------------------------------------------------*/
/* Stop collecting audio samples in buffers */
void Audio_stopCollect(Audio_configParams * audioConfig)
{
    Timer_A_stop(TIMER_A0_BASE);
    ADC12_B_disableConversions(ADC12_B_BASE, ADC12_B_COMPLETECONVERSION);
    DMA_disableTransfers(DMA_CHANNEL_0);
    DMA_disableInterrupt(DMA_CHANNEL_0);
}

/*--------------------------------------------------------------------------*/
/* Shut down the audio collection peripherals*/
void Audio_shutdownCollect(Audio_configParams * audioConfig)
{
    // Turn off preamp power
    MIC_POWER_PORT_OUT &= ~MIC_POWER_PIN;

    // Turn off preamp power
    GPIO_setOutputLowOnPin(GPIO_PORT_P4, GPIO_PIN1);

    // Disable the ADC
    ADC12_B_disable(ADC12_B_BASE);
}

/*--------------------------------------------------------------------------*/
/* Get pointer to active buffer with valid data ready for processing */
int16_t * Audio_getActiveBuffer(Audio_configParams * audioConfig)
{
    // Check if it is using only single or dual buffer
    if(audioConfig->audioBuffer2 != 0)
    {
        if(audioConfig->fillingBuffer1)return(audioConfig->audioBuffer2);
        else return(audioConfig->audioBuffer1);
    }
    else return(audioConfig->audioBuffer1);
}

/*--------------------------------------------------------------------------*/
/* Get pointer to the current buffer where data is currently written */
int16_t * Audio_getCurrentBuffer(Audio_configParams * audioConfig)
{
    // Check if it is using only single or dual buffer
    if(audioConfig->audioBuffer2 != 0)
    {
        if(audioConfig->fillingBuffer1)return(audioConfig->audioBuffer1);
        else return(audioConfig->audioBuffer2);
    }
    else return(audioConfig->audioBuffer1);
}

/*--------------------------------------------------------------------------*/
/* Indicate if an active buffer exists to be processed */
bool Audio_getActive(Audio_configParams * audioConfig)
{
    return(audioConfig->bufferActive);
}

/*--------------------------------------------------------------------------*/
/* Indicate done processing active buffer holding valid data */
void Audio_resetActive(Audio_configParams * audioConfig)
{
    audioConfig->bufferActive = false;
}

/*--------------------------------------------------------------------------*/
/* Get overflow status */
bool Audio_getOverflow(Audio_configParams * audioConfig)
{
    return(audioConfig->overflow);
}

/*--------------------------------------------------------------------------*/
/* Reset overflow status */
void Audio_resetOverflow(Audio_configParams * audioConfig)
{
    audioConfig->overflow = false;
}


