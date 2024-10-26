from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def C(n: int, k: int) -> float:
    """Sampling without replacement where order doesn't matter (Binomial Coefficient)
    """
        
    return math.comb(n, k)


class Card:
    
    def __init__(self, rank, suit):
        """Instantiates a Card object
        
        """
        self.rank = rank
        self.suit = suit
        self.card = (self.rank, self.suit)

class CardDeck:
    
    def __init__(self):
        """Instantiates a CardDeck object
        
        """
        self.card_deck = []
        self.create_deck()
        self.shuffle_deck()
    
    def create_deck(self) -> None:
        """Creates a deck of 52 cards, excluding Jokers
        as they are rarely used in recreational games like poker.
        
    
        """
        
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ["Hearts", "Clubs", "Spades", "Diamonds"]
        
        self.card_deck = []
        
        for suit in suits:
            for i in range(13):             #13 cards of each suit
                self.card_deck.append(Card(ranks[i], suit))
    
    def shuffle_deck(self) -> None:
        """Shuffles deck of 52 cards using the Fisher-Yates Algorithm.

        """
    
        deck = self.card_deck[:]
        shuffled_deck = []
    
        while len(deck) > 0:
            random_index = np.random.randint(0, len(deck))
            shuffled_deck.append(deck[random_index])
            del deck[random_index]
        self.card_deck = shuffled_deck
    
    def get_cards(self) -> list:
        """Returns a list of tuples, each tuple representing the attribute
        of a Card object
        
        """
        
        cards = []
        for i in range(len(self.card_deck)):
            card = self.card_deck[i]
            cards.append(card.card)
        return cards
    
class PokerHand:
    
    def __init__(self):
        """Instantiates a PokerHand object
        
        """
        self.hand = []
        self.draw_hand()
    
    def draw_hand(self) -> None:
        """Draws a hand, 5 cards, from the Card Deck using the Fisher-Yates 
        Algorithm
        
        """
        
        deck = CardDeck()
        self.hand = []
        for i in range(5):
            random_index = np.random.randint(0, len(deck.card_deck))
            self.hand.append(deck.card_deck[random_index])
            del deck.card_deck[random_index]
    
    def get_hand(self) -> list:
        """Returns a list of 5 tuples, each tuple representing an attribute of
        the Card object
        
        """
        hand = []
        for i in range(5): #Size of hand is always 5
            card = self.hand[i]
            hand.append(card.card)
            
        return hand
    
    def rank_in_hand(self, rank: str):
        """Returns True if inputted rank is in the hand, False otherwise
        
        """
        
        hand = self.get_hand()
        
        for card in hand:
            
            card_rank = card[0]
                
            if card_rank == rank:
                return True
                
        return False 
    
    def not_straight(self) -> bool:
        """Returns True if hand is not a straight, False otherwise
        
        """
        hand = self.get_hand()
        
        card_value = {}
        sorted_values = []
        
        i = 0
        
        for card in hand:
            
            rank = card[0]
            suit = card[1]
            
            i += 1
            
            if rank.isnumeric():
                card_value[rank] = int(rank)
            elif rank == 'J':
                card_value[rank] = 11
            elif rank == 'Q':
                card_value[rank] = 12
            elif rank == 'K':
                card_value[rank] = 13
            elif rank == 'A' and self.rank_in_hand('K'):
                card_value[rank] = 14
            elif rank == 'A' and self.rank_in_hand('2'):
                card_value[rank] = 1
                
        while card_value:
            
            min_value = min(card_value.values())
            sorted_values.append(min_value)
            rank = min(card_value, key = card_value.get)
            del card_value[rank]
        
        length = len(sorted_values)
        
        for j in range(length - 1):
            if sorted_values[j] + 1 != sorted_values[j + 1]:
                return True
            
        return False
    
    def is_high_card(self) -> bool:
        """Returns True if hand is a high card, False otherwise
        
        
        """
        
        hand = self.get_hand()
        
        i = 0
        
        for card in hand:
            
            rank = card[0]
            suit = card[1]
            
            suit_check = 1
            
            i += 1
            
            for next_card in hand[i:]:
                
                next_rank = next_card[0]
                next_suit = next_card[1]
                    
                if rank == next_rank:
                    return False
                    
                if suit == next_suit:
                    suit_check += 1
                        
                    if suit_check > 4:
                        return False
                    
        return self.not_straight()
    
    def is_pair(self) -> bool:
        """
        
        
        """
        hand = self.get_hand()
        
        rank_frequencies = {}
        pairs = 0
        
        for card in hand:
            
            rank = card[0]
            
            if rank in rank_frequencies:
                rank_frequencies[rank] += 1
            else:
                rank_frequencies[rank] = 1
                
        for frequency in rank_frequencies.values():
            if frequency > 2:
                return False
            elif frequency == 2:
                pairs += 1
                
        return pairs == 1
    
    def is_double_pair(self) -> bool:
        """
        
        """
        
        hand = self.get_hand()
        
        rank_frequencies = {}
        pairs = 0
        
        for card in hand:
            
            rank = card[0]
            
            if rank in rank_frequencies:
                rank_frequencies[rank] += 1
            else:
                rank_frequencies[rank] = 1
                
        for frequency in rank_frequencies.values():
            if frequency > 2:
                return False
            elif frequency == 2:
                pairs += 1
                
        return pairs == 2
    
    def three_of_a_kind(self) -> bool:
        """
        
        """
        
        hand = self.get_hand()
        
        rank_frequencies = {}
        
        rank_three = 0
        
        for card in hand:
            
            rank = card[0]
            
            if rank in rank_frequencies:
                rank_frequencies[rank] += 1
            else:
                rank_frequencies[rank] = 1
                
        for frequency in rank_frequencies.values():
            if frequency == 2:
                return False
            elif frequency == 3:
                rank_three += 1
        
        return rank_three == 1
    
    def is_straight(self) -> bool:
        """
        
        
        """
        
        hand = self.get_hand()
        
        suit_check = 1
        i = 0
        
        for card in hand:
            
            suit = card[1]
            
            i += 1
            
            for next_card in hand[i:]:
                
                if suit == next_card[1]:
                    
                    suit_check += 1
                    
            return not self.not_straight() and suit_check <= 4
                
    def is_flush(self) -> bool:
        """
        
        
        """
        
        hand = self.get_hand()
        
        suit1 = hand[0][1]
        i = 1
        
        for card in hand[i:]:
            if suit1 != card[1]:
                return False
            
        return self.not_straight()
    
    def is_full_house(self) -> bool:
        """
        
        
        """
        
        hand = self.get_hand()
        
        rank_frequencies = {}
        pairs = 0
        rank_three = 0
        
        for card in hand:
            
            rank = card[0]
            
            if rank in rank_frequencies:
                rank_frequencies[rank] += 1
            else:
                rank_frequencies[rank] = 1
                
        for frequency in rank_frequencies.values():
            if frequency > 3:
                return False
            elif frequency == 2:
                pairs += 1
            elif frequency == 3:
                rank_three += 1
                
        return pairs == 1 and rank_three == 1
    
    def four_of_a_kind(self) -> bool:
        """
        
        
        """
        
        hand = self.get_hand()
        
        rank_frequencies = {}
        rank_four = 0
        
        for card in hand:
            
            rank = card[0]
            
            if rank in rank_frequencies:
                rank_frequencies[rank] += 1
            else:
                rank_frequencies[rank] = 1
                
        for frequency in rank_frequencies.values():
            if frequency == 5:
                return False
            elif frequency == 4:
                rank_four += 1
                
        return rank_four == 1
    
    def is_straight_flush(self) -> bool:
        """
        
        """
        
        hand = self.get_hand()
        
        royal_ranks = '10JQKA'
        royal_check = 0
        suit1 = hand[0][1]
        
        for card in hand:
            if card[0] in royal_ranks:
                royal_check += 1
            if suit1 != card[1]:
                return False
            
        return not self.not_straight() and royal_check != 5
    
    def is_royal_flush(self) -> bool:
        """
        
        """
        
        hand = self.get_hand()
        
        royal_ranks = '10JQKA'
        i = 1
        suit1 = hand[0][1]
        
        for card in hand:
            if card[0] not in royal_ranks:
                return False
            if suit1 != card[1]:
                return False
        
        return True
    
    def hand_classification(self) -> str:
        """
        
        
        """ 
        
        if self.is_high_card():
            return "High Card"
        elif self.is_pair():
            return "Single Pair"
        elif self.is_double_pair():
            return "Double Pair"
        elif self.three_of_a_kind():
            return "Three of a Kind"
        elif self.is_straight():
            return "Straight"
        elif self.is_flush():
            return "Flush"
        elif self.is_full_house():
            return "Full House"
        elif self.four_of_a_kind():
            return "Four of a Kind"
        elif self.is_straight_flush():
            return "Straight Flush"
        elif self.is_royal_flush():
            return "Royal Flush"
    
    def hand_probability(self) -> float:
        """Returns the probability of drawing a poker hand
        
        >>> hand_probability(hand)
        
        
        """
        
        if self.is_high_card():
            return round(((C(13, 5) - 10) * (4 ** 5 - 4)) / C(52, 5), 7)
        
        elif self.is_pair():
            return round((C(13, 1) * C(4, 2) * C(12, 3) * 4 ** 3) / (C(52, 5)), 7)
        
        elif self.is_double_pair():
            return round((C(13, 2) * C(4, 2) ** 2 * C(11, 1) * 4) / C(52, 5), 7)
        
        elif self.three_of_a_kind():
            return round((C(13, 1) * C(4, 3) * C(12, 2) * 4 ** 2) / C(52, 5), 7)
        
        elif self.is_straight():
            return round((10 * 4 ** 5 - 10 * 4) / C(52, 5), 7) 
        #subtract straight flushes
        
        elif self.is_flush():
            
            return round(((C(13, 5) - 10) * 4) / C(52, 5), 7)
        #subtract straight flushes
        elif self.is_full_house():
            return round((C(13, 1) * C(4, 3) * C(12, 1) * C(4, 2)) / C(52, 5), 7)
            
        elif self.four_of_a_kind():
            return round((C(13, 1) * 12 * 4) / C(52, 5), 7) 
        
        elif self.is_straight_flush():
            return round((10 * 4 - 4) / C(52, 5), 7)
        
        elif self.is_royal_flush():
            return round(1 * 4 / C(52, 5), 7)
    
    def poker_hand_prob(self, poker_hand: str) -> float:
        """
        
        """
        
        if poker_hand == "High Card":
            return round(((C(13, 5) - 10) * (4 ** 5 - 4)) / C(52, 5), 7)
        
        elif poker_hand == "Single Pair":
            return round((C(13, 1) * C(4, 2) * C(12, 3) * 4 ** 3) / (C(52, 5)), 7)
        
        elif poker_hand == "Double Pair":
            return round((C(13, 2) * C(4, 2) ** 2 * C(11, 1) * 4) / C(52, 5), 7)
        
        elif poker_hand == "Three of a Kind":
            return round((C(13, 1) * C(4, 3) * C(12, 2) * 4 ** 2) / C(52, 5), 7)
        
        elif poker_hand == "Straight":
            return round((10 * 4 ** 5 - 10 * 4) / C(52, 5), 7) 
        #subtract straight flushes
        
        elif poker_hand == "Flush":
            
            return round(((C(13, 5) - 10) * 4) / C(52, 5), 7)
        #subtract straight flushes
        elif poker_hand == "Full House":
            return round((C(13, 1) * C(4, 3) * C(12, 1) * C(4, 2)) / C(52, 5), 7)
            
        elif poker_hand == "Four of a Kind":
            return round((C(13, 1) * 12 * 4) / C(52, 5), 7) 
        
        elif poker_hand == "Straight Flush":
            return round((10 * 4 - 4) / C(52, 5), 7)
        
        elif poker_hand == "Royal Flush":
            return round(1 * 4 / C(52, 5), 7)        
    
    
    def theoretical_prob_dict(self, poker_hands: list) -> dict:
        """
        
        """
        
        theoretical_prob_dict = {}
        
        for hand in poker_hands:
            theoretical_prob_dict[hand] = self.poker_hand_prob(hand)    
        
        return theoretical_prob_dict

    def prob_dict(self, hand_occurrences: dict, theoretical_dict: dict,
                  poker_hands: list) -> dict:
        """
        
        
        """
        
        prob_dict = {}
        
        i = 0
        
        hand_frequencies = []
        
        counter = 0
        
        for value in hand_occurrences.values():
            
            counter += value
            hand_frequencies.append(value)
        
        for value in theoretical_dict.values():
            empirical_value = round(hand_frequencies[i] / counter, 7)
            prob_dict[poker_hands[i]] = np.array([value, empirical_value])
            i += 1
            
        return prob_dict
    
    def simulate_hands(self, trials: int) -> list:
        """
        
        """
        hand_occurrences = {}
        prob_dict = {}
        
        data_all_trials = []
        
        poker_hands = ["High Card", "Single Pair", "Double Pair", 
                   "Three of a Kind", "Straight", "Flush", "Full House", 
                   "Four of a Kind", "Straight Flush", "Royal Flush"] 
        
        
        theoretical_prob_dict = self.theoretical_prob_dict(poker_hands)
        
        for i in range(10):
            
            hand_occurrences[poker_hands[i]] = 0
            prob_dict[poker_hands[i]] = 0
        
        for i in range(trials):

            poker_hand = self.hand_classification()
                
            hand_occurrences[poker_hand] += 1
            
            self.draw_hand()
        
        for j in range(10):
            
            theoretical = self.poker_hand_prob(poker_hands[j])
            empirical = round(hand_occurrences[poker_hands[j]] / trials, 7)
            prob_dict[poker_hands[j]] = np.array([theoretical, empirical])
            
        data_all_trials.append(hand_occurrences)
        data_all_trials.append(prob_dict)
        
        return data_all_trials   
        
    
    def simulation_data(self, n_trials: int, interval: int) -> list:
        """
            
        """
        
        all_simulation_data = []
        
        
        
        poker_hands = ["High Card", "Single Pair", "Double Pair", 
                   "Three of a Kind", "Straight", "Flush", "Full House", 
                   "Four of a Kind", "Straight Flush", "Royal Flush"] 
                

    
        for k in range(50):
            
            data = []
            
            hand_occurrences = {}
                        
            empirical_dict = {}
            
            theoretical_dict = {}            
            
            expected_occurrences = {}
                        
            variance_dict = {}
            
            empirical_values = {}
                        
            empirical_mean = {}
            
            empirical_range = {}
            
            high_deviation_intervals = {}
            
            hand_occurrence_avg = {}
            
            temp_occurrence_avg = {}
            
            absolute_difference_mean = {}
            
            interval_expected_occurrences = {}
            
            difference_variance_dict = {}
            
            difference_values = {}
            
            skewness = {}
            
            hand_occurrence_variance = {}
            
            hand_frequencies = {}
            
            counter = 0            
            
            
            for x in range(10):
                
                
                hand_occurrences[poker_hands[x]] = 0
                                
                empirical_mean[poker_hands[x]] = 0  
                
                empirical_values[poker_hands[x]] = []
                
                high_deviation_intervals[poker_hands[x]] = 0
                
                hand_occurrence_avg[poker_hands[x]] = 0
                
                temp_occurrence_avg[poker_hands[x]] = 0
                
                absolute_difference_mean[poker_hands[x]] = 0
                
                interval_expected_occurrences[poker_hands[x]] = 0
                
                difference_values[poker_hands[x]] = []
                
                hand_frequencies[poker_hands[x]] = []
                
                            
            for i in range(n_trials):
            
                        
                poker_hand = self.hand_classification()
            
                hand_occurrences[poker_hand] += 1
                
                temp_occurrence_avg[poker_hand] += 1
                            
                counter += 1
            
                
                if (i + 1) % interval == 0 or (i + 1) == n_trials:
                    
                    for m in range(10):
                        
                        theoretical = self.poker_hand_prob(poker_hands[m])
                        
                        empirical = round(hand_occurrences[poker_hands[m]] / counter, 7)
                        
                        if abs(theoretical - empirical) > 0.01:
                            
                            high_deviation_intervals[poker_hands[m]] += 1
                        
                        empirical_mean[poker_hands[m]] += empirical
                        
                        empirical_values[poker_hands[m]].append(empirical) 
                        
                        hand_occurrence_avg[poker_hands[m]] += temp_occurrence_avg[poker_hands[m]]
                        
                        hand_frequencies[poker_hands[m]].append(temp_occurrence_avg[poker_hands[m]])
                        
                        absolute_difference_mean[poker_hands[m]] += abs(theoretical - empirical)
                                                
                        difference_values[poker_hands[m]].append(abs(theoretical - empirical))
                        
                        temp_occurrence_avg[poker_hands[m]] = 0
                        
                        
                        if (i + 1) == n_trials:
                            
                            empirical_mean[poker_hands[m]] = empirical_mean[poker_hands[m]] / (n_trials / interval)
                            hand_occurrence_avg[poker_hands[m]] = hand_occurrence_avg[poker_hands[m]] / (n_trials / interval)
                            absolute_difference_mean[poker_hands[m]] = absolute_difference_mean[poker_hands[m]] / (n_trials / interval)
                        
                    for hand in poker_hands:
                        
                        variance = 0
                        
                        difference_variance = 0
                        
                        sum_values = 0
                        
                        frequencies = 0
                        
                        empirical_range[hand] = max(empirical_values[hand]) - min(empirical_values[hand])
                        
                        for value in empirical_values[hand]:
                            
                            variance += ((value - empirical_mean[hand]) ** 2)
                            
                        variance_dict[hand] = round(variance / (n_trials / interval), 7)
                        
                        for value in difference_values[hand]:
                            
                            difference_variance += ((value - absolute_difference_mean[hand]) ** 2)
                        
                        difference_variance_dict[hand] = round(difference_variance / (n_trials / interval), 7)
                        
                        for value in empirical_values[hand]:
                            
                            if variance_dict[hand] == 0:
                                skewness[hand] = 0
                            else:
                                
                                sum_values += (((value - empirical_mean[hand]) / math.sqrt(variance_dict[hand])) ** 3)
                            
                                skewness[hand] = round((sum_values * (n_trials / interval)) / (((n_trials / interval) - 1) * ((n_trials / interval) - 2)), 7)
                                
                        for value in hand_frequencies[hand]:
                            
                            frequencies += ((value - hand_occurrence_avg[hand]) ** 2)
                        
                        hand_occurrence_variance[hand] = round(frequencies / (n_trials / interval), 7)
                            
                            
                self.draw_hand()
                            
                
            for j in range(10):
                    
                theoretical = self.poker_hand_prob(poker_hands[j])
                    
                empirical = round(hand_occurrences[poker_hands[j]] / counter, 7)
                    
                empirical_dict[poker_hands[j]] = empirical
                    
                expected_occurrences[poker_hands[j]] = theoretical * n_trials
                
                interval_expected_occurrences[poker_hands[j]] = theoretical * interval
                
                theoretical_dict[poker_hands[j]] = theoretical
                
                
            data.append(hand_occurrences)
            data.append(theoretical_dict)
            data.append(empirical_dict)
            data.append(expected_occurrences)
            data.append(empirical_mean)
            data.append(variance_dict)
            data.append(empirical_range)
            data.append(high_deviation_intervals)
            data.append(hand_occurrence_avg)
            data.append(absolute_difference_mean)
            data.append(interval_expected_occurrences)
            data.append(difference_variance_dict)
            data.append(skewness)
            data.append(hand_occurrence_variance)
            
            all_simulation_data.append(data)
        
        
        return all_simulation_data
        
    

def comparison_dataframe(data_all_trials: list):
    """
                          
    """
    
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    
    
    columns = list(data_all_trials[0].keys()) #Creates a list of column headers
    
    dataframe = pd.DataFrame(index = ["Hand Occurrences", "Theoretical Probability", "Empirical Probability"], columns = columns)
        
    occurrences = []
    
    theoretical = []
    
    empirical = []
    
    for value in data_all_trials[0].values():
        occurrences.append(value)
        
    dataframe.loc["Hand Occurrences"] = occurrences
    
    for value in data_all_trials[1].values():
        theoretical.append(value[0])
        empirical.append(value[1])
    
    dataframe.loc["Theoretical Probability"] = theoretical
    dataframe.loc["Empirical Probability"] = empirical
    
    
    return dataframe

def theoretical_vs_empirical_graph(dataframe: pd.DataFrame):
    """
    
    """

    width = 0.4 #width of each bar
    
    values = np.arange(10)
    
    plt.bar(values - width / 2, dataframe.loc["Theoretical Probability"], width, label = "Theoretical", color = 'blue')
    plt.bar(values + width / 2, dataframe.loc["Empirical Probability"], width, label = "Empirical", color = 'red')
    #width / 2 ensures the x ticks are centered between both bars
    
    
    plt.legend()
    
    plt.xlabel("Poker Hands")
    plt.ylabel("Probabilities")
    
    plt.xticks(values, dataframe.columns)
    
    plt.title("Empirical vs Theoretical Probability of Poker Hands")
    
    plt.show()
    
def simulation_dataframe(all_simulation_data: list):
    """
    
    """
        
    poker_hands = ["High Card", "Single Pair", "Double Pair", 
                   "Three of a Kind", "Straight", "Flush", "Full House", 
                   "Four of a Kind", "Straight Flush", "Royal Flush"]    
    
    rows = []
    
    for i in range(50):
        num = i + 1
        rows.append("Simulation " + str(num)) 
    
    columns = []
    
    for hand in poker_hands:
        
        columns.append(hand + " Hand Occurrences")
        columns.append(hand + " Theoretical Probability")        
        columns.append(hand + " Empirical Probability")
        columns.append(hand + " Expected Occurrences")
        columns.append(hand + " Empirical Probability Mean")
        columns.append(hand + " Empirical Probability Variance")
        columns.append(hand + " Absolute Empirical Max and Min Probability Range")
        columns.append(hand + " High Deviation Intervals")
        columns.append(hand + " Hand Occurrences Mean")
        columns.append(hand + " Absolute Theoretical and Empirical Probability Difference Mean")
        columns.append(hand + " Expected Hand Occurrences Per Interval")
        columns.append(hand + " Absolute Theoretical and Empirical Probability Variance") 
        columns.append(hand + " Skewness")  
        columns.append(hand + " Hand Occurrence Variance") 
        
    columns.append("Number of Trials")
        
    
    df = pd.DataFrame(index = rows, columns = columns)
        
    
    for i in range(50):
        num = i + 1
        sim = all_simulation_data[i]
        all_values = []        
        
        for hand in poker_hands:
            all_values.append(sim[0][hand])
            all_values.append(sim[1][hand])
            all_values.append(sim[2][hand])
            all_values.append(sim[3][hand])
            all_values.append(sim[4][hand])
            all_values.append(sim[5][hand])
            all_values.append(sim[6][hand])
            all_values.append(sim[7][hand])
            all_values.append(sim[8][hand])
            all_values.append(sim[9][hand])
            all_values.append(sim[10][hand])
            all_values.append(sim[11][hand])
            all_values.append(sim[12][hand])
            all_values.append(sim[13][hand])
            
        all_values.append(10000)    
        df.loc["Simulation " + str(num)] = all_values
        num += 1    
        
    return df

def simulation_csv(df: pd.DataFrame):
    """
    
    
    """
    df.to_csv("simulation_data.csv", index = True)
    
    return "simulation_data.csv"


def predict(csv_file: str, hand: str):
    
    df = pd.read_csv(csv_file)
    
    if hand == "High Card" or hand == "Single Pair":
    
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " High Deviation Intervals", hand + " Hand Occurrences Mean", hand + " Absolute Theoretical and Empirical Probability Difference Mean", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Skewness", "Number of Trials"]]
    
    elif hand == "Double Pair" or hand == "Three of a Kind" or hand == "Straight":
        
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " High Deviation Intervals", hand + " Hand Occurrences Mean", hand + " Absolute Theoretical and Empirical Probability Difference Mean", hand + " Expected Hand Occurrences Per Interval", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Hand Occurrence Variance", hand + " Skewness"]]
    
    elif hand == "Flush":
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " Expected Hand Occurrences Per Interval", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Hand Occurrence Variance"]]
    
    elif hand == "Full House":
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " High Deviation Intervals", hand + " Hand Occurrences Mean", hand + " Absolute Theoretical and Empirical Probability Difference Mean", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Skewness", "Number of Trials"]]
    
    elif hand == "Four of a Kind":
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " High Deviation Intervals", hand + " Hand Occurrences Mean", hand + " Absolute Theoretical and Empirical Probability Difference Mean", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Skewness", "Number of Trials"]]
    
    else:
        data = df[[hand + " Theoretical Probability", hand +  " Empirical Probability Mean", hand + " Empirical Probability Variance", hand + " Absolute Empirical Max and Min Probability Range", hand + " High Deviation Intervals", hand + " Hand Occurrences Mean", hand + " Absolute Theoretical and Empirical Probability Difference Mean", hand + " Absolute Theoretical and Empirical Probability Variance", hand + " Skewness", hand + " Hand Occurrence Variance"]]
            
    
    predict = hand + " Empirical Probability Variance"
    
    x = np.array(data.drop([predict], axis = 1))  # 1 implies dropping columns, 0 implies dropping rows
    y = np.array(data[predict])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    
    x_test_scaled = scaler.transform(x_test)
        
    linear = linear_model.LinearRegression()
    linear.fit(x_train_scaled, y_train)
        
    predictions = linear.predict(x_test_scaled)
    residuals = y_test - predictions    
        
    accuracy = linear.score(x_test_scaled, y_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
        
    print("Model Accuracy: " +  str(accuracy))
    print("Mean Squared Error: " +  str(mse))
    print("Mean Absolute Error: " +  str(mae))
    print("Root Mean Squared Error: " +  str(rmse))
    
    plt.figure()    
    plt.scatter(predictions, residuals, color = "blue")
    plt.axhline(y = 0, color = "black", linestyle = "--", linewidth = 1)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Predicted Values vs Residuals")
    
    plt.figure()    
    plt.scatter(y_test, predictions, color = "red")
    plt.axline((0, 0), slope = 1, color = "black", linestyle = "--", linewidth = 1)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actual Values")
        
    plt.show()    
    
    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])
    