```
use to track leetcode progress
```
# Leetcode

- Problem Solving
    
    [](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=711860&page=1&authorid=4353)
    
    - zigzag sequence
        
        ![Untitled](Leetcode%20cad4a2bcb8714e09a8da667e6a795b65/Untitled.png)
        
        ```python
        def findZigZagSequence(a, n):
        #first sort the list
            a.sort()
            mid = int((n + 1)/2) -1
        #swap mid with largest 
            a[mid], a[n-1] = a[n-1], a[mid]
        # first half is already sorted by increasing order, so just need to deal with 
        # the second half, compare number and swap if st <=ed as after sorting, number at st is larger than 
        # number at ed and we want decreasing order
            st = mid + 1
            ed = n - 2
            while(st <= ed):
                a[st], a[ed] = a[ed], a[st]
                st = st + 1
                ed = ed - 1
        
            for i in range (n):
                if i == n-1:
                    print(a[i])
                else:
                    print(a[i], end = ' ')
            return
        ```
        
    - find co-prime
        
        ```python
        find co-prime
        
        def gcd(a, b):
            # Define the while loop as described
            while b != 0:
                a, b = b, a%b   
            # Complete the return statement
            return a
            
        # Create a list of tuples defining pairs of coprime numbers
        coprimes = [(i, j) for i in list1 
                           for j in list2 if gcd(i, j) == 1]
        print(coprimes)
        ```
        
    - [weighted sampling](https://pynative.com/python-weighted-random-choices-with-probability/)
        
        You are given n numbers as well as probabilities p_0, p_1, ... , p_{n - 1}, which sum up to 1, how would you generate one of the n numbers according to the specified probabilities?
        
        For example, if the numbers are 3, 5, 7, 11, and the probabilities are 9/18, 6/18, 2/18, 1/18, then 100000 calls to your program, 3 should appear roughly 500000 times,
        5 should appear roughly 333333 times, 7 should appear roughly 111111 times, and 11 should appear roughly 55555 times.
        
        ```python
        # weighted random sample with replacement
        # return k sized list of elements
        random.choices(population, weights = None, cum_weights = None, k=1)
        
        import random
        
        # we specified head and tail of a coin in string
        coin = "HT"
        # Execute 3 times to verify we are getting 6 or more heads in every 10 spins
        for i in range(3):
            print(random.choices(coin, cum_weights=(0.61, 1.00), k=10))
        ```
        
    - random number generator
        
        ```python
        import random
        random.randint(0,9)
        ```
        
    - String Compression - easy
        
        Implement a method to perform basic string compression using the counts of repeated characters. 
        
        For example, the string aabcccccddd would become a2b1c5d3. If the "compressed" string would not become smaller than the original string, your method should return the original string.
        
        You can assume the string has only lowercase letters. (a-z)
        
        ```python
        def string_compress(s):
            result = ""
            cnt = 1
           
            for i in range(len(s) - 1):
                if s[i] == s[i + 1]:
                    cnt += 1
                else:
                    result += s[i] + str(cnt)
                    cnt = 1
                    
            result += s[i] + str(cnt)
           
            if len(result) >= len(s):
                return s
            return result
        
        if __name__ == "__main__":
            assert string_compress("aabcccccddd") == "a2b1c5d3"
            assert string_compress("abcde") == "abcde"
        ```
        
    - String Compression - medium
        
        Given an array of characters `chars`, compress it using the following algorithm:
        
        Begin with an empty string `s`. For each group of **consecutive repeating characters** in `chars`:
        
        - If the group's length is `1`, append the character to `s`.
        - Otherwise, append the character followed by the group's length.
        
        The compressed string `s` **should not be returned separately**, but instead, be stored **in the input character array `chars`**. Note that group lengths that are `10` or longer will be split into multiple characters in `chars`.
        
        After you are done **modifying the input array**, return *the new length of the array*.
        
        You must write an algorithm that uses only constant extra space.
        
        ```python
        class Solution:
            def compress(self, chars: List[str]) -> int:
                read = 0 
                write = 0
                
                while read < len(chars):
        #             initial char
                    char = chars[read]
        #     reset count to 0
                    count = 0
                    while read< len(chars) and chars[read] == char:
                        count +=1
                        read +=1
                            
                    chars[write] = char     
                    write +=1
                    if count > 1:
                        for i in str(count):
                            chars[write] = i
                            write +=1
                return write
        ```
        
    - Maximun Subarray
        
        Given an array of integers, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
        
        ```python
        Example:
        Input: [-2,1,-3,4,-1,2,1,-5,4],
        Output: 6
        Explanation: [4,-1,2,1] has the largest sum = 6.
        
        # optimized brute force --> can start with brute force and move to optimized
        Class solution:
        		def maxSubArray(self, nums: list[int]) ->int:
        # use negative infinity,not 0--> it's possible that only negative numbers in the array
        					max_sum = -math.inf
        					for i in range(len(nums)):
        							cur_subarray = 0
        							for j in range(i, len(nums)):
        									cur_subarray += nums[j]
        									max_sum = max(max_sum, cur_subarray)
        		return max_sum
        
        # dynamic programming
        
        Class solution:
            def maxSubArray(self, nums: list[int]) ->int:
                max_until_i = max_sum = nums[0]
                for i, num in enumerate(nums):
                    max_until_i = max(max_until_i + num, num)
                    max_sum = max(max_sum, max_until_i)
        ```
        
    - Smallest difference of two arrays
        
        Given two arrays of integers, compute the pair of values with the smallest non-negative difference.
        
        ```python
        Example:
        Input : A[] = {1, 3, 15, 11, 2}
                B[] = {23, 127, 235, 19, 8} 
        Output : 3  
        That is, the pair (11, 8) 
        
        Input : A[] = {l0, 5, 40}
                B[] = {50, 90, 80} 
        Output : 10
        That is, the pair (40, 50)
        
        Class solution:
        	def minDiff(self, a1, a2):
        			a1_sort = sorted(a1)
        			a2_sort = sorted(a2)
        			p1 = p2 = 0
        			min_diff = float(inf)
        			while p1 < len(a1_sort) and p2 < len(a2_sort):
        						diff = a1_sort[p1] - a2_sort[p2]
        						min_diff = min(min_diff, abs(diff))
        						if diff > 0:
        									p2 +=1
        						else:
        									p1 +=1
        			return min_diff
        ```
        
    - Best Time to Buy and Sell Stock (leetcode 121)
        
        You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.
        
        You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.
        
        Return *the maximum profit you can achieve from this transaction*. If you cannot achieve any profit, return `0`.
        
        ```python
        # key is to always track min purchase + max profit
        class Solution:
            def maxProfit(self, prices: List[int]) -> int:
                max_profit = 0
                min_purchase = prices[0]
                for i, price in enumerate(prices):
                    profit = price - min_purchase
                    max_profit = max(profit, max_profit)
                    min_purchase = min(price, min_purchase)
                return max_profit
        ```
        
    - Game of life
        
        According to [Wikipedia's article](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life): "The **Game of Life**, also known simply as **Life**, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."
        
        The board is made up of an `m x n` grid of cells, where each cell has an initial state: **live** (represented by a `1`) or **dead** (represented by a `0`). Each cell interacts with its [eight neighbors](https://en.wikipedia.org/wiki/Moore_neighborhood) (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
        
        1. Any live cell with fewer than two live neighbors dies as if caused by under-population.
        2. Any live cell with two or three live neighbors lives on to the next generation.
        3. Any live cell with more than three live neighbors dies, as if by over-population.
        4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
        
        The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the `m x n` grid `board`, return *the next state*.
        
        ![Untitled](Leetcode%20cad4a2bcb8714e09a8da667e6a795b65/Untitled%201.png)
        
        - code
            
            ```python
            class Solution:
                def gameOfLife(self, board: List[List[int]]) -> None:
                    """
                    Do not return anything, modify board in-place instead.
                    """
                    #         get rows and cols from nested list
                    rows = len(board)
                    cols = len(board[0])
                    
                    def sum_neighbors(row, col):
                        
                        neighbors_location = [(-1,-1),(-1,0),(1,-1),(0,-1),(0,1),(-1,1),(1,0),(1,1)]
                        
                        live_ct = 0
                        for neighbor_location in neighbors_location:
                            neighbor_row = row + neighbor_location[0]
                            neighbor_col = col + neighbor_location[1]
                            if (neighbor_row >= 0 and neighbor_row< rows) and (neighbor_col< cols and neighbor_col >= 0) and abs(board[neighbor_row][neighbor_col]) ==1:
                                # abs 
                                live_ct +=1
                                print(row, col, live_ct)
                                
                        return  live_ct
                    # loop through rows and cols
                    for row in range(rows):
                        for col in range(cols):
                            # cell = board[row][col]
                            if board[row][col] == 1 and(sum_neighbors(row, col) < 2 or sum_neighbors(row, col) > 3):
                                board[row][col] = -1    
                                # if sum_neighbors == 2 or  sum_neighbors == 3:
                                #     cell == 1
                            if board[row][col] == 0 and(sum_neighbors(row, col) ==3):
                                board[row][col] = 2
                                
                                
                    # map back 
                    for row in range(rows):
                        for col in range(cols):
                            if board[row][col] > 0 :
                                board[row][col] = 1
                            if board[row][col] < 0 :
                                board[row][col] = 0
            ```
            
- Two Pointers Technique
    
    Two pointers is really an easy and effective technique that is typically used for searching pairs in a sorted array.
    
    Given a sorted array A (sorted in ascending order), having N integers, find if there exists any pair of elements (A[i], A[j]) such that their sum is equal to X.
    
    Now let’s see how the two-pointer technique works. We take two pointers, one representing the first element and other representing the last element of the array, and then we add the values kept at both the pointers. If their sum is smaller than X then we shift the left pointer to right or if their sum is greater than X then we shift the right pointer to left, in order to get closer to the sum. We keep moving the pointers until we get the sum as X.
