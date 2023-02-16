/*
 * global.h
 *
 *  Created on: Nov 4, 2022
 *      Author: mahathir
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <msp430.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <driverlib.h>
#include <DSPLib.h>

#include <QmathLib.h>
#include <IQmathLib.h>

#include "audio/collect_audio.h"

#define __SYSTEM_FREQUENCY_MHZ__ 8000000

extern Audio_configParams gAudioConfig;

void init_infer(void);
uint16_t infer(void);

#endif /* GLOBAL_H_ */
