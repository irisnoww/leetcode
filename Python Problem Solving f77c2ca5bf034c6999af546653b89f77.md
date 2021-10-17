# Python Problem Solving

[13个好用到起飞的Python技巧！](https://mp.weixin.qq.com/s/CWGqU0LNu0ogkecY1LB7MQ)

[Pandas](https://www.notion.so/dac15a1de3414f60b34e05c3c521f8e4)

- basics
    - how to print
        
        ```python
        print("geeks", end =" ")
        
        #print in new line , using \n
        print('{0}\n{1}\n{2}\n'.format((a+b),(a-b),(a*b)))
        
        #print 2 decimals
        avg_mark = "{:.2f}".format((tot_mark/3))
        
        #'\n' ---> new line
        print('\n'.join([i for i in sorted([x[0] for x in list if x[1]==sec_last])]))
        
        The newline character
        \n
        \n is still one character
        stuff = 'X\nY'
        print (stuff)
        X
        Y
        
        read 2 new lines sperately and print them together
        print("Here are {} {}".format(input(),input()))
        
        #print % number
        print('{:.1%}'.format(prob))
        ```
        
    - open file
        
        ```python
        using open
        handle = open(filename, mode)
        'r' read
        'w' write
        
        The newline character
        \n
        \n is still one character
        stuff = 'X\nY'
        print (stuff)
        X
        Y
        len(stuff)
        3
        
        fhand = open('file')
        for line in file:
        	if line.startswith():
        			print 
        #rm white space
        line.strip()
        
        ### reading an "incorrect" CSV to dataframe having a variable number of columns/tokens 
        import pandas as pd
        
        df = pd.read_csv('Test.csv', header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        # ... do some modifications with df
        ### end of code
        ```
        
    - list comprehension
        
        #list comprehension
        #use [ ]
        print([[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i + J +k != n])
        
    - string split and join
        
        ```python
        #string spilt and join
        def split_and_join(line):
            #write your code here
            a=line.split(" ")
            a="-".join(a)
            return a
        
        # split string and join in a new line
        def wrap(string, max_width)
        # use \n to join new line, slicing (start, end(not included))
        	return "\n".join(string[i:i+max_width] for i in range(0, len(string), max_width)
        ```
        
    - functional programming
        - map()
            
            > map(func, iterable)
            > 
            - knowledge
                - apply a transformation function to each item in an iterable and transform them into a new iterable.
                - Since map() is written in C and is highly optimized, its internal implied loop can be more efficient than a regular Python for loop. This is one advantage of using map().+ memory consumption ++
                - map() return map object, need to call list() to return list
            - application
                
                ```python
                >>> str_nums = ["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]
                # func WITHOUT ()
                >>> int_nums = map(int, str_nums)
                >>> int_nums
                <map object at 0x7fb2c7e34c70>
                
                >>> list(int_nums)
                [4, 8, 6, 5, 3, 2, 8, 9, 2, 5]
                
                >>> str_nums
                ["4", "8", "6", "5", "3", "2", "8", "9", "2", "5"]
                
                # combined with lambda
                squared = map(lambda x: x**2, numbers)
                
                # input e.g.: 1 2
                a = list(map(int, input().split()))
                b = list(map(int, input().split()))
                ```
                
        - filter()
        - reduce
    - open file json txt
- functions application
    - removing duplicates from string
        - order not matters
            
            ```python
            "".join(set(string))
            ```
            
        - order matter
            
            ```python
            "".join(dict.fromkeys(string))
            ```
            
            ```python
            "".join(sorted(set(string), key = string.index))
            ```
            
        - sorted & set & join
            
            The sorted() function returns a sorted list from the items in an iterable.
            sorted(iterable, key=None, reverse=False)
            if reverse = True, sorted in descending order
            
            ```python
            #print the second max value from array
            print(sorted(set(arr),reverse = True)[1])
            
            #set
            #sorted set --> remove duplicates in list and then sort
            sec_last = sorted(set([i[1] for i in list]))[1]
            
            # .join() with lists
            numList = ['1', '2', '3', '4']
            separator = ', '
            print(separator.join(numList))
            ```
            
    - find the most common letter in letters
        - Counter
            
            ```python
            #output a dictionary with Key: value
            #value is count(key)
            from collections import Counter
            N = int(input())
            #split()
            Shoe_list = Counter(list(map(int, input().split())))
            Customer = int(input())
            Money = 0
            for i in range(Customer):
                Selected_size, value = map(int, input().split())
            		#if > 0 / true
                if Shoe_list[Selected_size]:
                    Money += value
                    Shoe_list[Selected_size]-=1
                    
            print(Money)
            
            -- note: most_common() return a list
            ```
            
        - Map
            
            ```python
            #map
            #pass every item in iterable into function
            Return Value from map()
            The map() function applies a given to function to each item of an iterable 
            and returns a list of the results.
            
            The returned value from map() (map object) can then be passed to 
            functions like list() (to create a list), set() (to create a set) and so on.
            
            map(function, iterable)
            ```
            
        
        ```python
        if __name__ == '__main__':
            s = sorted(input())
            from collections import Counter
        #most_common return a list of (item: occurence)
            letters = Counter(s).most_common(3)
            for letter in letters:
                print(letter[0],letter[1])
        
        ```
        
    - itertools
- Problem Solving
    
    [](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=711860&page=1&authorid=4353)
    
    - zigzag sequence
        
        ![Untitled](Python%20Problem%20Solving%20f77c2ca5bf034c6999af546653b89f77/Untitled.png)
        
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
    - string compression
        
        Implement a method to perform basic string compression using the counts of repeated characters. For example, the string aabcccccddd would become a2b1c5d3. If the "compressed" string would not become smaller than the original string, your method should return the original string.
        
        You can assume the string has only lowercase letters. (a-z)
        
        ```python
        
        ```
        
    - largest sum subarray
        
        Given an array of integers, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
        Example:
        Input: [-2,1,-3,4,-1,2,1,-5,4],
        Output: 6
        Explanation: [4,-1,2,1] has the largest sum = 6.
        
    - Smallest difference of two arrays
        
        Given two arrays of integers, compute the pair of values with the smallest non-negative difference.
        
        ```python
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
        
        
    - Dutch National Flag Problem
        
        此题即Leetcode 75 Sort Colors
        
        Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
        We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
        
        Example:
        Input: nums = [2,0,2,1,1,0]
        Output: [0,0,1,1,2,2]
        
    - rotate matrix
        
        You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise and anticlockwise).
        
        You have to rotate the image in-place.
        
    - Reservoir sampling
        
        Reservoir sampling is a family of randomized algorithms for randomly choosing a sample of k items from a list S containing n items, where n is either a very large or unknown number. Typically, n is too large to fit the whole list into main memory.
        
    - Count Primes (leetcode 204)
        
        Count the number of prime numbers less than a non-negative number, n.
        
        Input: n = 10
        Output: 4
        Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
        
    - Most Frequent Words
        
        Given a file that contains the text of a document and an integer k, return a list of the top k most frequent words.
        
    - Find median in an unsorted array
        
        
    - Word Break (leetcode 139)
    - Longest Palindromic Substring (leetcode 5)
    - Even Odd Split
    - Merge two sorted arrays
    - Intersection of Two Arrays (Leetcode 349)
    - Fibonacci Number (Leedcode 509)
- Visualization
- Data Wrangling
    - apply lambda
        
        ```python
        # 1）udf with lambda
        def myfunc():
        	pass
        # axis = 1 apply to each row
        df['new_col'] = df['col'].apply(lambda x: myfunc(x), axis = 1)
        
        # 2） lambda with if else 
        df['new_col'] = df['col'].apply(lambda x: 1 if x>0 else 0)
        
        # 3) result['business_type'] = result['business_name'].apply(
        lambda x: 'school' if 'school' in x.lower()
            else 'restaurant' if 'restaurant' in x.lower()
            else 'cafe' if 'cafe' in x.lower() or 'coffee' in x.lower()
        else 'other')
        
        ```
        
    - sort dataframe
        
        ```python
        df.sort_values(by = 'col', ascending=False)
        ```
        
    
    ```python
    # Create function for lambda
    Example 1:
    def myfunc(position, result_check):
        if result_check==True and position<=3:
            myvalue=5
        elif result_check==True and position<=5:
            myvalue=4
        elif result_check==True and position<=10:
            myvalue=3
        elif result_check==True and position>10:
            myvalue=2
        else:
            myvalue=1
    	return myvalue
    
    fb_search_results['rating'] = fb_search_results.apply(lambda x: myfunc(x['position'], x['result_check']), axis=1)
    
    Example 2:
    facebook_posts['is_spam'] = facebook_posts['is_spam'].apply(lambda x: 1 if x == True else 0)
    
    Example 3:
    fb_search_results['result_check'] = fb_search_results.apply(lambda x: x.query.lower() in x.notes.lower(), axis=1)   
    
    Example 4:
    result['business_type'] = result['business_name'].apply(
    lambda x: 'school' if 'school' in x.lower()
        else 'restaurant' if 'restaurant' in x.lower()
        else 'cafe' if 'cafe' in x.lower() or 'coffee' in x.lower()
    else 'other')
    
    # create new column with dict
    df.replace({"col1": di})
    
    # Order by asc/ desc
    df.sort_values(by='col1', ascending=False)
    
    # Not in
    ~customers['id'].isin(name_ls)
    
    # In
    customers['id'].isin(name_ls)
    
    # Or
    (df['A'] == 3) | (df['B'] == 7)
    
    # Top
    Df.head(5)
    
    # Null
    Df.col.Isna()
    Df.col.notnull()
    
    # Unstack
    DataFrame.unstack(level=- 1, fill_value=None)
    Pivot a level of the (necessarily hierarchical) index labels.
    Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.
    If the index is not a MultiIndex, the output will be a Series (the analogue of stack when the columns are not a MultiIndex).
    Parameters
    Level int, str, or list of these, default -1 (last level)
    Level(s) of index to unstack, can pass level name.
    fill_value int, str or dict
    Replace NaN with this value if the unstack produces missing values.
    
    # Contains
    [Series.str.contains(pat, case=True, flags=0, na=None, regex=True)](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html)
    Test if pattern or regex is contained within a string of a Series or Index.
    Return boolean Series or Index based on whether a given pattern or regex is contained within a string of a Series or Index.
    
    Example 1: 
    # case: if true, case sensitive
    airbnb_search_details['amenities'].str.contains('beach', case=False)
    
    # Offset
    df.col1.shift(-1)
    
    # Row number
    Df.reset_index()
    
    # Fill NA
    Df.fillna(0)
    
    # Idxmin
    DataFrame.idxmin(axis=0, skipna=True)
    Return index of first occurrence of minimum over requested axis.
    NA/null values are excluded.
    
    		Parameters
    		Axis {0 or ‘index’, 1 or ‘columns’}, default 0
    		The axis to use. 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise.
    		skipnabool, default True-baidu 1point3acres
    		Exclude NA/null values. If an entire row/column is NA, the result will be NA.
    
    # Replace 
    relative_variance_hr.replace({"PWCOUNTY_name": county_name}).copy()
    
    # Rank
    df1['rank'] = df1['msg_count'].rank(ascending=False)
    - method{‘average’, ‘min’, ‘max’, ‘first’, ‘dense’}, default ‘average’
    How to rank the group of records that have the same value (i.e. ties):
    •        average: average rank of the group
    •        min: lowest rank in the group
    •        max: highest rank in the group
    •        first: ranks assigned in order they appear in the array
    •        dense: like ‘min’, but rank always increases by 1 between groups
    - pct, default False
    Whether or not to display the returned rankings in percentile form
    
    # Append
    Df1.append([df2, df3])
    
    # Drop_duplicates
    DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    Return DataFrame with duplicate rows removed.
    Considering certain columns is optional. Indexes, including time indexes are ignored.
    Parameters
    Subset column label or sequence of labels, optional. 
    Only consider certain columns for identifying duplicates, by default use all of the columns.
    Keep {‘first’, ‘last’, False}, default ‘first’. From 1point 3acres bbs
    Determines which duplicates (if any) to keep. - first : Drop duplicates except for the first occurrence. - last : Drop duplicates except for the last occurrence. - False : Drop all duplicates.
    Inplace bool, default False
    Whether to drop duplicates in place or to return a copy.
    ignore_index bool, default False
    If True, the resulting axis will be labeled 0, 1, …, n - 1.
    New in version 1.0.0.
    Example 1: df.drop_duplicates(subset=['brand', 'style'], keep='last')
    
    # Groupby
    After groupby, you can add  'max', 'mean', 'median', 'min', 'count', ‘prod’ (i.e. taking product 3*4), 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'fillna', 'filter', 'nunique', 'pct_change', 'quantile', 'rank',  'sum', ‘size’, ‘nlargest(n)’
    # agg
    Example 1: result = result.groupby('post_date').agg({'is_spam': ['sum', 'count']}).reset_index()
    Example 2: df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
    Example 3: count_facilities = los_angeles_restaurant_health_inspections.groupby(["facility_zip"])['facility_id'].agg(no_facilities='nunique', no_inspections='count').reset_index()
    
    # apply: use it with lambda function
    # diff
    Periods: int, default 1
    Periods to shift for calculating difference, accepts negative values.
    Axis: {0 or ‘index’, 1 or ‘columns’}, default 0
    Take difference over rows (0) or columns (1).
    
    # transform
    DataFrameGroupBy.transform(func, *args, engine=None, engine_kwargs=None, **kwargs)
    Call function producing a like-indexed DataFrame on each group and return a DataFrame having the same indexes as the original object filled with the transformed values
    Parameters
    F function
    Function to apply to each group.
    Can also accept a Numba JIT function with engine='numba' specified.. check 1point3acres for more.
    If the 'numba' engine is chosen, the function must be a user defined function with values and index as the first and second arguments respectively in the function signature. Each group’s index will be passed to the user defined function and optionally available for use.
    Changed in version 1.1.0.
    *args
    Positional arguments to pass to func.
    Engine str, default None
    •        'cython' : Runs the function through C-extensions from cython.
    •        'numba' : Runs the function through JIT compiled code from numba.
    •        None : Defaults to 'cython' or globally setting compute.use_numba
    engine_kwargs dict, default None
    •        For 'cython' engine, there are no accepted engine_kwargs
    •        For 'numba' engine, the engine can accept nopython, nogil and parallel dictionary keys. The values must either be True or False. The default engine_kwargs for the 'numba' engine is {'nopython': True, 'nogil': False, 'parallel': False} and will be applied to the function
    New in version 1.1.0.
    **kwargs
    Keyword arguments to be passed into func.
    
    Pivot_table
    pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
    Create a spreadsheet-style pivot table as a DataFrame.
    The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the index and columns of the result DataFrame.
    Parameters
    Data DataFrame
    Values column to aggregate, optional
    Index column, Grouper, array, or list of the previous
    If an array is passed, it must be the same length as the data. The list can contain any of the other types (except list). Keys to group by on the pivot table index. If an array is passed, it is being used as the same manner as column values.
    Columns column, Grouper, array, or list of the previous
    If an array is passed, it must be the same length as the data. The list can contain any of the other types (except list). Keys to group by on the pivot table column. If an array is passed, it is being used as the same manner as column values.
    Aggfunc function, list of functions, dict, default numpy.mean
    If list of functions passed, the resulting pivot table will have hierarchical columns whose top level are the function names (inferred from the function objects themselves) If dict is passed, the key is column to aggregate and value is function or list of functions.
    fill_value scalar, default None
    Value to replace missing values with (in the resulting pivot table, after aggregation).
    Margins bool, default False
    Add all row / columns (e.g. for subtotal / grand totals).
    Dropna bool, default True
    Do not include columns whose entries are all NaN.
    margins_name str, default ‘All’
    Name of the row / column that will contain the totals when margins is True.
    Observed bool, default False
    This only applies if any of the groupers are Categoricals. If True: only show observed values for categorical groupers. If False: show all values for categorical groupers.
    Changed in version 0.25.0.
    Example 1: pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value=0)
    Example 2: pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean, 'E': [min, max, np.mean]})
    Rename multindex table
    Example 1:
    pivot_df.columns = pivot_df.columns.droplevel(0)
    pivot_df.columns.name = None
    pivot_df.columns= pivot_df.columns.astype(str)
    pivot_df = pivot_df.reset_index()
    result = pivot_df.rename(columns = {'Entire home/apt':'apt_count','Private room':'private_count','Shared room':'shared_count'})
    
    # Merge
    pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
             left_index=False, right_index=False, sort=True,
             suffixes=['_x', '_y'], copy=True, indicator=False,
             validate=None)
    sort bool, default False
    Sort the join keys lexicographically in the result DataFrame. If False, the order of the join keys depends on the join type (how keyword).
    Copy bool, default True
    If False, avoid copy if possible.
    Indicator bool or str, default False
    If True, adds a column to the output DataFrame called “_merge” with information on the source of each row.
    Validate str, optional
    If specified, checks if merge is of specified type.
    •        “one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.
    •        “one_to_many” or “1:m”: check if merge keys are unique in left dataset.
    •        “many_to_one” or “m:1”: check if merge keys are unique in right dataset.
    •        “many_to_many” or “m:m”: allowed, but does not result in checks.
    
    String manipulation
    # string extract by location
    Df.col.str[6, 2]
    # string extract by content
    Df.col.str.contains() 
    # add string
    ‘text1’+’text2’
    # split
    Pd.DataFrame(Df.col.Split(‘,’).tolist())
    Calculation Operation-baidu 1point3acres
    # round
    df2['pay_emp'] = round(df2['pay_emp'], 2)
    # others
    abs(), np.sqrt(), np.std(), power: 5**3 = 125
    
    Data Type  
    Df.col.astype({'col1': 'int32'})
    
    Calculation Date
    # convert datetime
    from datetime import datetime
    datetime.now().year
    # format
    pd.to_datetime(df['activity_date'], format='%Y-%m-%d' ).dt.strftime('%Y-%m-%d')
    # delta
    fb_comments_count['created_at'] >= pd.to_datetime('2020-02-10') - timedelta(days=30)
    # find year/month/day
    X.INCDTTM = pd.to_datetime(X.INCDTTM)
    X.INCDTTM.dt.year
    # between
    merged['date'].between('2020-03-01', '2020-03-31')]
    
    Rename
    count = count.rename({0: 'count'}, axis='columns').copy()
    
    Regax
    result['business_name'].replace('[^a-zA-Z0-9 ]','',regex=True)
    
    Tips and Tricks
    # remember to sort see if we need to include name/ product that has zero records.
    ```
    
- bucket & bin your data
    
    If you want equal distribution of the items in your bins, use qcut . If you want to define your own numeric bin ranges, then use cut .
    
    ```python
    qcut()
    df['quantile_ex_4'] = pd.qcut(df['ext price'],
                                q=[0, .2, .4, .6, .8, 1],
                                labels=False,
                                precision=0)
    df.head()
    # use to specifically define the bin edges
    cut()
    # use np.linespace to create equally distributed range
    pd.cut(df['ext price'], bins=np.linspace(0, 200000, 9))
    # add label to cut
    cut_labels_4 = ['silver', 'gold', 'platinum', 'diamond']
    cut_bins = [0, 70000, 100000, 130000, 200000]
    df['cut_ex1'] = pd.cut(df['ext price'], bins=cut_bins, labels=cut_labels_4)
    ```
    
- 
- Two Pointers Technique
    
    Two pointers is really an easy and effective technique that is typically used for searching pairs in a sorted array.
    
    Given a sorted array A (sorted in ascending order), having N integers, find if there exists any pair of elements (A[i], A[j]) such that their sum is equal to X.
    
    Now let’s see how the two-pointer technique works. We take two pointers, one representing the first element and other representing the last element of the array, and then we add the values kept at both the pointers. If their sum is smaller than X then we shift the left pointer to right or if their sum is greater than X then we shift the right pointer to left, in order to get closer to the sum. We keep moving the pointers until we get the sum as X.