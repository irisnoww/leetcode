```
twosum

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].

# uses a hashmap to find complement values, and therefore achieves \mathcal{O}(N)O(N) time complexity.
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, num in enumerate(nums):
            if target - num in d:
                return [i, d[target - num]]
            d[num] = i
			
____________________________
twosumII

Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= first < second <= numbers.length.

Return the indices of the two numbers, index1 and index2, as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.

# uses the two pointers pattern and also has (N)O(N) time complexity for a sorted array. 
# We can use this approach for any array if we sort it first, which bumps the time complexity to (n\log{n})O(nlogn).

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        p1 = 0
        p2 = len(numbers) -1
        while p1 < len(numbers):
            sum_num = numbers[p1] + numbers[p2]
            if sum_num < target:
                p1 += 1
            elif sum_num > target:
                p2 -=1
            else:
                return [p1 + 1, p2+1]



____________________________
two sum less than k
Given an array nums of integers and integer k, return the maximum sum such that there exists i < j with nums[i] + nums[j] = sum and sum < k. 
If no i, j exist satisfying this equation, return -1.

#     two-pointers
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        nums.sort()
        r = 0
        l = len(nums) - 1
        max_sum = -1
        while r < l:
            cur_sum = nums[r] + nums[l]
            if cur_sum < k:
                max_sum = max(max_sum, cur_sum)
                r +=1
            else:
                l -= 1
        return max_sum
____________________________
 three sum
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, 
and nums[i] + nums[j] + nums[k] == 0.

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
      
        result = []
        for i in range(len(nums)):
           
            if nums[i] > 0:
                break
            if i == 0 or nums[i] != nums[i-1]:
                p1 = i + 1
                p2 = len(nums)-1
                while p1 < p2:
                    if nums[p1] + nums[p2] > -nums[i] :
                        p2 -=1
                    elif nums[p1] + nums[p2] < -nums[i]:
                        p1 +=1
                    else:
                        result.append([nums[p1], nums[p2], nums[i]])
                    
                        p1 +=1
                        p2 -+1
                        while p1 < p2 and nums[p1] == nums[p1-1]:
                            p1 +=1
          
        return result
