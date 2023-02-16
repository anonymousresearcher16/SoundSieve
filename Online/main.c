#include "global.h"

/**
 * main.c
 */


#define MemoryReadPin GPIO_PORT_P3, GPIO_PIN7  // LOW: execute program, HIGH: read memory
#define MODE GPIO_PORT_P7, GPIO_PIN1

void initClock(void);
void initGPIO(void);


int main(void)
{
	WDTCTL = WDTPW | WDTHOLD;	// stop watchdog timer
	initClock();
	initGPIO();

	if(GPIO_getInputPinValue(MemoryReadPin) == GPIO_INPUT_PIN_LOW )    // if LOW, execute program, otherwise, do not execute any code
	{
	    init_infer();
	    infer();
}
	
	__bis_SR_register(LPM3_bits + GIE);
}


void initClock()
{
    // Configure one FRAM waitstate as required by the device datasheet for MCLK
    // operation beyond 8MHz _before_ configuring the clock system.
    //FRCTL0 = FRCTLPW | NWAITS_1;  // uncomment this line if >8MHz

    //Startup clock system with max DCO setting ~1MHz
    CSCTL0_H = CSKEY_H;
    //CSCTL1 = DCOFSEL_0; //Set DC0 to 1 MHz
    CSCTL1 = DCOFSEL_6; // Set DCO, DCOFSEL_6 = 8MHz for 8MHz sampling frequency, comment previous line and uncomment this line

    //MCLK is used as clock source for the CPU.
    //SMCLK is usually a high frequency clock and it is used for peripheral modules (e.g. Timers, serial communication modules, ...)
    //ACLK is usually a 32kHz crystal clock. It is used for peripheral modules that require a low-frequency clock (e.g. real-time-clock,  ...)

    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;

    CSCTL4 = LFXTOFF | HFXTOFF; //save power

    CSCTL0_H = 0;
}


void initGPIO()
{
    // Configure GPIO
    // save energy
    PADIR=0xffff; PAOUT=0x0000; // Ports 1 and 2
    PBDIR=0xffff; PBOUT=0x0000; // Ports 3 and 4
    PCDIR=0xffff; PCOUT=0x0000; // Ports 5 and 6
    PDDIR=0xffff; PDOUT=0x0000; // Ports 7 and 8
    PJDIR=0xff; PJOUT=0x00;     // Port J

    /* for reading memory with UniFlash without executing the code
     *  if this pin is LOW, then execute program
     *  if this pin is HIGH, do not execute, will use UniFlash to read memory values
    */
    GPIO_setAsInputPin(MemoryReadPin);  // #define MemoryReadPin

    // Mode of low-power mic. If High the mic is in low-power wake up mode waiting for an event to occur. If low, mic is in active mode
    GPIO_setAsOutputPin(MODE);
    GPIO_setOutputLowOnPin(MODE);

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

}

//******************************************************************************
// DMA interrupt service routine
//******************************************************************************
#pragma vector=DMA_VECTOR
__interrupt void dmaIsrHandler(void)
{
    switch(__even_in_range(DMAIV, DMAIV_DMA5IFG))
    {
    case DMAIV_DMA0IFG:
        // Switch the audio buffer for ping-pong buffer
        Audio_switchBuffer(&gAudioConfig);
        // Exit low power mode on wake-up
        __bic_SR_register_on_exit(LPM3_bits);
        break;
    case DMAIV_DMA1IFG:
        // Disable the dma transfer
        DMA_disableTransfers(DMA_CHANNEL_1);

        // Clear the DMA request
        DMA1CTL &= ~(DMAREQ);

        // Disable DMA channel 1 interrupt
        DMA_disableInterrupt(DMA_CHANNEL_1);

        // Exit low power mode on wake-up
        __bic_SR_register_on_exit(LPM3_bits);
        break;
    case DMAIV_DMA2IFG:
        // Disable the dma transfer
        DMA_disableTransfers(DMA_CHANNEL_2);

        // Clear the DMA request
        DMA2CTL &= ~(DMAREQ);

        // Disable DMA channel 2 interrupt
        DMA_disableInterrupt(DMA_CHANNEL_2);

        // Exit low power mode on wake-up
        __bic_SR_register_on_exit(LPM3_bits);
        break;
    case DMAIV_DMA3IFG:
        // Disable the dma transfer
        DMA_disableTransfers(DMA_CHANNEL_3);

        // Clear the DMA request
        DMA3CTL &= ~(DMAREQ);

        // Disable DMA channel 3 interrupt
        DMA_disableInterrupt(DMA_CHANNEL_3);

        // Exit low power mode on wake-up
        __bic_SR_register_on_exit(LPM3_bits);
        break;
    case DMAIV_DMA4IFG: break;
    case DMAIV_DMA5IFG: break;
    default: break;
    }
}

#pragma vector=ADC12_VECTOR
__interrupt void ADC12_ISR(void)
{
    switch(__even_in_range(ADC12IV, ADC12IV_ADC12RDYIFG))
    {
    case ADC12IV_NONE:        break;        // Vector  0:  No interrupt
    case ADC12IV_ADC12OVIFG:  break;        // Vector  2:  ADC12MEMx Overflow
    case ADC12IV_ADC12TOVIFG: break;        // Vector  4:  Conversion time overflow
    case ADC12IV_ADC12HIIFG:  break;        // Vector  6:  ADC12HI
    case ADC12IV_ADC12LOIFG:  break;        // Vector  8:  ADC12LO
    case ADC12IV_ADC12INIFG:  break;        // Vector 10:  ADC12IN
    case ADC12IV_ADC12IFG0:   break;        // Vector 12:  ADC12MEM0 Interrupt
    case ADC12IV_ADC12IFG1:   break;        // Vector 14:  ADC12MEM1
    case ADC12IV_ADC12IFG2:   break;        // Vector 16:  ADC12MEM2
    case ADC12IV_ADC12IFG3:   break;        // Vector 18:  ADC12MEM3
    case ADC12IV_ADC12IFG4:   break;        // Vector 20:  ADC12MEM4
    case ADC12IV_ADC12IFG5:   break;        // Vector 22:  ADC12MEM5
    case ADC12IV_ADC12IFG6:   break;        // Vector 24:  ADC12MEM6
    case ADC12IV_ADC12IFG7:   break;        // Vector 26:  ADC12MEM7
    case ADC12IV_ADC12IFG8:   break;        // Vector 28:  ADC12MEM8
    case ADC12IV_ADC12IFG9:   break;        // Vector 30:  ADC12MEM9
    case ADC12IV_ADC12IFG10:  break;        // Vector 32:  ADC12MEM10
    case ADC12IV_ADC12IFG11:  break;        // Vector 34:  ADC12MEM11
    case ADC12IV_ADC12IFG12:  break;        // Vector 36:  ADC12MEM12
    case ADC12IV_ADC12IFG13:  break;        // Vector 38:  ADC12MEM13
    case ADC12IV_ADC12IFG14:  break;        // Vector 40:  ADC12MEM14
    case ADC12IV_ADC12IFG15:  break;        // Vector 42:  ADC12MEM15
    case ADC12IV_ADC12IFG16:  break;        // Vector 44:  ADC12MEM16
    case ADC12IV_ADC12IFG17:  break;        // Vector 46:  ADC12MEM17
    case ADC12IV_ADC12IFG18:  break;        // Vector 48:  ADC12MEM18
    case ADC12IV_ADC12IFG19:  break;        // Vector 50:  ADC12MEM19
    case ADC12IV_ADC12IFG20:  break;        // Vector 52:  ADC12MEM20
    case ADC12IV_ADC12IFG21:  break;        // Vector 54:  ADC12MEM21
    case ADC12IV_ADC12IFG22:  break;        // Vector 56:  ADC12MEM22
    case ADC12IV_ADC12IFG23:  break;        // Vector 58:  ADC12MEM23
    case ADC12IV_ADC12IFG24:  break;        // Vector 60:  ADC12MEM24
    case ADC12IV_ADC12IFG25:  break;        // Vector 62:  ADC12MEM25
    case ADC12IV_ADC12IFG26:  break;        // Vector 64:  ADC12MEM26
    case ADC12IV_ADC12IFG27:  break;        // Vector 66:  ADC12MEM27
    case ADC12IV_ADC12IFG28:  break;        // Vector 68:  ADC12MEM28
    case ADC12IV_ADC12IFG29:  break;        // Vector 70:  ADC12MEM29
    case ADC12IV_ADC12IFG30:  break;        // Vector 72:  ADC12MEM30
    case ADC12IV_ADC12IFG31:  break;        // Vector 74:  ADC12MEM31
    case ADC12IV_ADC12RDYIFG: break;        // Vector 76:  ADC12RDY
    default: break;
    }
}
