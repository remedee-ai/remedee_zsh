import asyncio
import os
import sys
import time
import unittest

sys.path.append(os.path.abspath("."))

from remedee.common.llm.llm import LLM

class LlmParallelTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = LLM(model="4o", cache=False, instance="test")

    async def _run_parallel_queries(self, num_queries):
        # Define async function for executing a single query
        async def query_task(task_id):
            start_time = time.time()
            response, input_tokens = await self.llm.query_simple("Write a poem with 6 times 4 rows! Then translate it to German.")
            end_time = time.time()
            execution_time = end_time - start_time
            return execution_time

        # Measure execution time for a single query using the query_task function
        execution_time_single = await query_task(0)
        print(f"Single query execution time: {execution_time_single:.2f} seconds")

        # Run two queries in parallel
        tasks = [asyncio.create_task(query_task(i)) for i in range(1, num_queries+1)]

        # Measure total execution time for parallel tasks
        start_time_parallel = time.time()
        execution_times = await asyncio.gather(*tasks)
        end_time_parallel = time.time()
        total_parallel_time = end_time_parallel - start_time_parallel

        # print results
        print(f"Total parallel execution time: {total_parallel_time:.2f} seconds")
        for i, execution_time in enumerate(execution_times):
            print(f"Execution time for query {i+1}: {execution_time:.2f} seconds")
            
        # check results
        for i, execution_time in enumerate(execution_times):
            self.assertLess(execution_time, execution_time_single * 1.2)

    # async def test_two_parallel(self):
    #     await self._run_parallel_queries(2)

    # async def test_two_parallel(self):
    #     await self._run_parallel_queries(4)

    async def test_two_parallel(self):
        await self._run_parallel_queries(8)

# To run the test directly (if not using a test runner)
if __name__ == "__main__":
    unittest.main()
