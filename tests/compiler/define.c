/* define the hist level too. assume <= 32. */
#define CONFIG_DATA_LOG2 13
/* dont touch */
#define CONFIG_DATA_DIM (1 << CONFIG_DATA_LOG2)
#define CONFIG_BLOCK_DIM (CONFIG_DATA_DIM / 8)

