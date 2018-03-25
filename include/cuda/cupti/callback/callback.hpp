#ifndef CUDA_CUPTI_CALLBACK_CALLBACK_HPP
#define CUDA_CUPTI_CALLBACK_CALLBACK_HPP

#include <cupti.h>

void CUPTIAPI cuptiCallbackFunction(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    CUpti_CallbackData *cbdata);

#endif